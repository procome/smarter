# pylint: disable=broad-exception-caught
"""
Dental appointment scheduling tool for the Smarter platform.

Overview
--------
Provides a single LLM-callable tool function, dental_appointment(), that
handles four scheduling actions via a required 'action' parameter:

  lookup  – check whether a patient already has an appointment on a given date
  book    – schedule a new appointment (returns BOOKED or next available slot)
  confirm – confirm a SCHEDULED appointment by appointment_id
  cancel  – cancel an appointment by appointment_id

Access is restricted to authenticated clinic staff. The two-layer authorization
model (chatbot YAML + in-function guard) mirrors the pattern used throughout the
Smarter platform.

Dependencies
------------
- pymysql
- pydantic
- django-redis (for session caching)

Environment Variables
---------------------
- DENTAL_DB_NAME         — clinic scheduling database name (default: "dental_clinic")
- DENTAL_DB_TABLE_PREFIX — optional prefix prepended to table names (default: "")

Signals
-------
- llm_tool_presented
- llm_tool_requested
- llm_tool_responded
"""

import logging
import os
import re
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Iterator, Optional

import pymysql
import pymysql.cursors
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from smarter.common.enum import SmarterEnum
from smarter.common.helpers.console_helpers import formatted_text
from smarter.common.utils import is_authenticated_request
from smarter.lib import json
from smarter.lib.django import waffle
from smarter.lib.django.waffle import SmarterWaffleSwitches
from smarter.lib.logging import WaffleSwitchedLoggerWrapper

from ..signals import llm_tool_requested, llm_tool_responded


# ── Logging ───────────────────────────────────────────────────────────────────

# pylint: disable=W0613
def should_log(level):
    return waffle.switch_is_active(SmarterWaffleSwitches.PROMPT_LOGGING)


base_logger = logging.getLogger(__name__)
logger = WaffleSwitchedLoggerWrapper(base_logger, should_log)
logger_prefix = formatted_text(__name__)


# ── Enums ─────────────────────────────────────────────────────────────────────

class DentalAction(SmarterEnum):
    """Supported action values for the dental_appointment tool."""
    LOOKUP  = "lookup"
    BOOK    = "book"
    CONFIRM = "confirm"
    CANCEL  = "cancel"


class AppointmentType(SmarterEnum):
    """Supported appointment types."""
    CHECKUP      = "CHECKUP"
    CLEANING     = "CLEANING"
    XRAY         = "XRAY"
    CONSULTATION = "CONSULTATION"
    EMERGENCY    = "EMERGENCY"


class AppointmentStatus(SmarterEnum):
    """Status values returned to the LLM."""
    FOUND            = "FOUND"            # lookup found an existing appointment
    NOT_FOUND        = "NOT_FOUND"        # lookup: no appointment on that date
    BOOKED           = "BOOKED"           # book: new appointment inserted
    UNAVAILABLE      = "UNAVAILABLE"      # book: slot taken, next available returned
    CONFIRMED        = "CONFIRMED"
    CANCELLED        = "CANCELLED"
    PENDING_APPROVAL = "PENDING_APPROVAL" # book: EMERGENCY requires HITL pre-approval


# ── Custom Exception ──────────────────────────────────────────────────────────

class DentalAppointmentError(Exception):
    """Custom exception for dental appointment tool errors."""


# ── Staff-Only Authorization Helper ──────────────────────────────────────────
# Mirrors is_staff check in SmarterAdminWebView.dispatch() (smarter/lib/django/views.py)

def _require_staff_user(request) -> Optional[str]:
    """
    Verify the caller is an authenticated staff member.

    Returns
    -------
    Optional[str]
        An error string if access is denied, or None if authorized.
    """
    if request is None:
        logger.warning("%s Tool called with no request context — access denied.", logger_prefix)
        return "Access denied. This tool is restricted to authenticated clinic staff."

    user = getattr(request, "user", None)

    if not is_authenticated_request(request) or not getattr(user, "is_authenticated", False):
        logger.warning("%s Unauthenticated access attempt to dental_appointment.", logger_prefix)
        return "Access denied. This tool is restricted to authenticated clinic staff."

    if not getattr(user, "is_staff", False):
        logger.warning(
            "%s Non-staff user '%s' denied access.",
            logger_prefix,
            getattr(user, "username", "unknown"),
        )
        return "Access denied. This tool is restricted to authenticated clinic staff."

    logger.debug("%s Staff user '%s' authorized.", logger_prefix, user.username)
    return None


# ── Database Constants ────────────────────────────────────────────────────────

CLINIC_OPEN_HOUR    = 8
CLINIC_CLOSE_HOUR   = 17
SLOT_DURATION_MIN   = 30
BOOKING_WINDOW_DAYS = 7


# ── Database Helper Class ─────────────────────────────────────────────────────
# Modelled on the Stackademy DatabaseConnection class (agentic-ai-workflow/app/database.py)

class DentalSchedulingDB:
    """
    Manages MySQL connections and queries for the clinic scheduling database.
    Uses the same pymysql / context-manager pattern as the Stackademy plugin.
    """

    def __init__(self):
        self.host         = os.environ.get("MYSQL_HOST", "localhost")
        self.port         = int(os.environ.get("MYSQL_PORT", 3306))
        self.user         = os.environ.get("MYSQL_USERNAME", "")
        self.password     = os.environ.get("MYSQL_PASSWORD", "")
        self.database     = os.environ.get("DENTAL_DB_NAME", "dental_clinic")
        self.table_prefix = os.environ.get("DENTAL_DB_TABLE_PREFIX", "")
        self.charset      = "utf8mb4"

    @property
    def _patients(self) -> str:
        return f"{self.table_prefix}patients"

    @property
    def _appointments(self) -> str:
        return f"{self.table_prefix}appointments"

    @contextmanager
    def get_cursor(self) -> Iterator[pymysql.cursors.DictCursor]:
        """Context manager: open connection, yield DictCursor, commit or rollback."""
        connection = None
        try:
            connection = pymysql.connect(
                host=self.host, port=self.port, user=self.user,
                password=self.password, database=self.database,
                charset=self.charset, cursorclass=pymysql.cursors.DictCursor,
                autocommit=False,
            )
            cursor = connection.cursor()
            yield cursor
            connection.commit()
        except Exception as e:
            if connection:
                connection.rollback()
            raise e
        finally:
            if connection:
                connection.close()

    def lookup_patient_appointment(self, patient_name: str, appt_date: str) -> Optional[dict]:
        """Return the active appointment for patient_name on appt_date, or None."""
        query = f"""
            SELECT a.appointment_id, a.appointment_time, a.appointment_type, a.status
            FROM   {self._appointments} a
            JOIN   {self._patients}     p ON a.patient_id = p.patient_id
            WHERE  p.patient_name    = %s
              AND  a.appointment_date = %s
              AND  a.status NOT IN ('CANCELLED')
            LIMIT 1
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (patient_name, appt_date))
            return cursor.fetchone()

    def is_slot_available(self, appt_date: str, appt_time: str) -> bool:
        """Return True if no non-cancelled appointment exists for the given date/time."""
        query = f"""
            SELECT appointment_id FROM {self._appointments}
            WHERE appointment_date = %s
              AND appointment_time = %s
              AND status != 'CANCELLED'
            LIMIT 1
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (appt_date, appt_time))
            return cursor.fetchone() is None

    def find_next_available(self, from_date: str, from_time: str) -> Optional[dict]:
        """
        Search forward up to BOOKING_WINDOW_DAYS for the next open 30-minute slot.
        Skips weekends. Returns {"date": "YYYY-MM-DD", "time": "HH:MM"} or None.
        """
        start_dt = datetime.strptime(f"{from_date} {from_time}", "%Y-%m-%d %H:%M")
        start_dt += timedelta(minutes=SLOT_DURATION_MIN)
        max_slots = BOOKING_WINDOW_DAYS * (CLINIC_CLOSE_HOUR - CLINIC_OPEN_HOUR) * (60 // SLOT_DURATION_MIN)
        for _ in range(max_slots):
            if start_dt.hour >= CLINIC_CLOSE_HOUR:
                start_dt = start_dt.replace(hour=CLINIC_OPEN_HOUR, minute=0) + timedelta(days=1)
            if start_dt.weekday() >= 5:
                start_dt += timedelta(days=1)
                start_dt = start_dt.replace(hour=CLINIC_OPEN_HOUR, minute=0)
            d = start_dt.strftime("%Y-%m-%d")
            t = start_dt.strftime("%H:%M")
            if self.is_slot_available(d, t):
                return {"date": d, "time": t}
            start_dt += timedelta(minutes=SLOT_DURATION_MIN)
        return None

    def get_or_create_patient(self, patient_name: str) -> int:
        """Return existing patient_id or insert a new patient row and return its id."""
        with self.get_cursor() as cursor:
            cursor.execute(
                f"SELECT patient_id FROM {self._patients} WHERE patient_name = %s LIMIT 1",
                (patient_name,),
            )
            row = cursor.fetchone()
            if row:
                return row["patient_id"]
            cursor.execute(f"INSERT INTO {self._patients} (patient_name) VALUES (%s)", (patient_name,))
            return cursor.lastrowid

    def book_appointment(self, patient_id: int, appt_date: str,
                         appt_time: str, appt_type: str) -> int:
        """Insert appointment row. Returns the new appointment_id."""
        query = f"""
            INSERT INTO {self._appointments}
                (patient_id, appointment_date, appointment_time, appointment_type, status)
            VALUES (%s, %s, %s, %s, 'SCHEDULED')
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (patient_id, appt_date, appt_time, appt_type))
            return cursor.lastrowid

    def get_appointment(self, appointment_id: int) -> Optional[dict]:
        """Return appointment row with patient name, or None if not found."""
        query = f"""
            SELECT a.appointment_id, p.patient_name,
                   a.appointment_date, a.appointment_time,
                   a.appointment_type, a.status
            FROM   {self._appointments} a
            JOIN   {self._patients}     p ON a.patient_id = p.patient_id
            WHERE  a.appointment_id = %s
            LIMIT 1
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (appointment_id,))
            return cursor.fetchone()

    def update_appointment_status(self, appointment_id: int, status: str) -> None:
        """Update the status column of an appointment row."""
        with self.get_cursor() as cursor:
            cursor.execute(
                f"UPDATE {self._appointments} SET status = %s WHERE appointment_id = %s",
                (status, appointment_id),
            )


dental_db = DentalSchedulingDB()


# ── Validation Helpers ────────────────────────────────────────────────────────

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def _validate_date(value: str) -> Optional[str]:
    """Return an error string, or None if the date is valid and in the future."""
    if not DATE_RE.match(value):
        return f"Invalid date format. Expected YYYY-MM-DD, received: '{value}'."
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return f"'{value}' is not a valid calendar date."
    if parsed < date.today():
        return "Requested date is in the past. Please provide a future date."
    return None


def _validate_time(value: str) -> Optional[str]:
    """Return an error string, or None if the time is within clinic hours."""
    if not TIME_RE.match(value):
        return f"Invalid time format. Expected HH:MM (24-hour), received: '{value}'."
    hour = int(value.split(":")[0])
    if not (CLINIC_OPEN_HOUR <= hour < CLINIC_CLOSE_HOUR):
        return f"Requested time is outside clinic hours ({CLINIC_OPEN_HOUR:02d}:00–{CLINIC_CLOSE_HOUR:02d}:00)."
    return None


# ── Main Tool Function ────────────────────────────────────────────────────────
# Modelled on get_current_weather() in smarter/apps/prompt/functions/function_weather.py

def dental_appointment(
    tool_call: ChatCompletionMessageToolCall,
    request=None,
) -> list:
    """
    Unified dental appointment tool: lookup, book, confirm, or cancel.

    The 'action' argument selects the operation. Required arguments vary by action:
      lookup  — patient_name, date
      book    — patient_name, date, time, appointment_type (optional)
      confirm — appointment_id
      cancel  — appointment_id, reason (optional)

    Access is restricted to authenticated staff members.

    Parameters
    ----------
    tool_call : ChatCompletionMessageToolCall
        The OpenAI tool call object containing function name and arguments.
    request : HttpRequest, optional
        The Django request forwarded by the Smarter platform.

    Returns
    -------
    list
        A JSON-compatible list containing one result dict or error dict.
    """
    # ── 0. Staff-only authorization ───────────────────────────────────────────
    auth_error = _require_staff_user(request)
    if auth_error:
        return [{"error": auth_error}]

    # ── 1. Parse arguments ────────────────────────────────────────────────────
    arguments = {}
    if tool_call and tool_call.function and tool_call.function.arguments:
        if isinstance(tool_call.function.arguments, str):
            try:
                arguments = json.loads(tool_call.function.arguments)
                logger.debug("%s Parsed arguments: %s", logger_prefix, json.dumps(arguments, indent=2))
            except Exception as e:
                logger.error("%s Error parsing arguments JSON: %s", logger_prefix, e)
                return [{"error": f"Invalid arguments JSON: {e}"}]
        else:
            arguments = tool_call.function.arguments

    # ── 2. Validate action ────────────────────────────────────────────────────
    action = arguments.get("action")
    if not action:
        return [{"error": "Missing required parameter: 'action'. Supported: lookup, book, confirm, cancel."}]
    if action not in DentalAction.all():
        return [{"error": f"Invalid action '{action}'. Supported: {', '.join(DentalAction.all())}."}]

    # ── 3. Fire request signal ────────────────────────────────────────────────
    llm_tool_requested.send(
        sender=dental_appointment,
        tool_call=tool_call.model_dump(),
        action=action,
        arguments=arguments,
    )

    # ── 4. Dispatch to action handler ─────────────────────────────────────────
    try:
        if action == DentalAction.LOOKUP:
            result = _handle_lookup(arguments)
        elif action == DentalAction.BOOK:
            result = _handle_book(arguments)
        elif action == DentalAction.CONFIRM:
            result = _handle_confirm(arguments)
        elif action == DentalAction.CANCEL:
            result = _handle_cancel(arguments)
    except pymysql.Error as e:
        logger.error("%s Database error: %s", logger_prefix, e)
        return [{"error": f"Database error: {e}"}]
    except Exception as e:
        logger.error("%s Unexpected error: %s", logger_prefix, e)
        return [{"error": f"Unexpected error: {e}"}]

    # ── 5. Fire response signal ───────────────────────────────────────────────
    llm_tool_responded.send(
        sender=dental_appointment,
        tool_call=tool_call.model_dump(),
        tool_response=result,
    )
    return [result]


# ── Action Handlers ───────────────────────────────────────────────────────────

def _handle_lookup(args: dict) -> dict:
    """Check whether a patient has an existing appointment on a given date."""
    patient_name = args.get("patient_name", "").strip()
    appt_date    = args.get("date", "").strip()

    if not patient_name:
        return {"error": "Missing required parameter: 'patient_name'."}
    if not appt_date:
        return {"error": "Missing required parameter: 'date'."}
    err = _validate_date(appt_date)
    if err:
        return {"error": err}

    row = dental_db.lookup_patient_appointment(patient_name, appt_date)
    if row:
        return {
            "status":           AppointmentStatus.FOUND,
            "patient_name":     patient_name,
            "date":             appt_date,
            "appointment_id":   row["appointment_id"],
            "appointment_time": str(row["appointment_time"]),
            "appointment_type": row["appointment_type"],
            "appointment_status": row["status"],
            "message": (
                f"{patient_name} has a {row['appointment_type']} "
                f"at {row['appointment_time']} on {appt_date} (status: {row['status']})."
            ),
        }
    return {
        "status":       AppointmentStatus.NOT_FOUND,
        "patient_name": patient_name,
        "date":         appt_date,
        "message":      f"No active appointment found for {patient_name} on {appt_date}.",
    }


def _handle_book(args: dict) -> dict:
    """Book a new appointment, or return the next available slot."""
    patient_name     = args.get("patient_name", "").strip()
    appt_date        = args.get("date", "").strip()
    appt_time        = args.get("time", "").strip()
    appointment_type = args.get("appointment_type", AppointmentType.CHECKUP)

    for name, value in [("patient_name", patient_name), ("date", appt_date), ("time", appt_time)]:
        if not value:
            return {"error": f"Missing required parameter: '{name}'."}

    err = _validate_date(appt_date)
    if err:
        return {"error": err}
    err = _validate_time(appt_time)
    if err:
        return {"error": err}
    if appointment_type not in AppointmentType.all():
        return {"error": f"Invalid appointment_type '{appointment_type}'. Supported: {', '.join(AppointmentType.all())}."}

    # TODO: DAP30 Module 6 — replace this guard with a real HITL pre-approval workflow
    if appointment_type == AppointmentType.EMERGENCY:
        return {
            "status":           AppointmentStatus.PENDING_APPROVAL,
            "patient_name":     patient_name,
            "appointment_type": appointment_type,
            "date":             appt_date,
            "time":             appt_time,
            "message": (
                "EMERGENCY appointments require supervisor pre-approval. "
                "Please contact the clinic supervisor to complete this booking manually."
            ),
        }

    # Check for duplicate booking first
    existing = dental_db.lookup_patient_appointment(patient_name, appt_date)
    if existing:
        return {
            "error": (
                f"{patient_name} already has a {existing['appointment_type']} "
                f"at {existing['appointment_time']} on {appt_date}. "
                f"Use action='cancel' to cancel it first, or choose a different date."
            ),
        }

    if dental_db.is_slot_available(appt_date, appt_time):
        patient_id = dental_db.get_or_create_patient(patient_name)
        appt_id    = dental_db.book_appointment(patient_id, appt_date, appt_time, appointment_type)
        return {
            "status":           AppointmentStatus.BOOKED,
            "appointment_id":   appt_id,
            "patient_name":     patient_name,
            "appointment_type": appointment_type,
            "date":             appt_date,
            "time":             appt_time,
        }

    next_slot = dental_db.find_next_available(appt_date, appt_time)
    if next_slot:
        return {
            "status":               AppointmentStatus.UNAVAILABLE,
            "requested_date":       appt_date,
            "requested_time":       appt_time,
            "next_available_date":  next_slot["date"],
            "next_available_time":  next_slot["time"],
            "message": (
                f"The {appt_time} slot on {appt_date} is unavailable. "
                f"Next available: {next_slot['date']} at {next_slot['time']}."
            ),
        }
    return {"error": f"No available slots found in the next {BOOKING_WINDOW_DAYS} days. Please call the clinic directly."}


def _handle_confirm(args: dict) -> dict:
    """Confirm a SCHEDULED appointment by appointment_id."""
    appointment_id = args.get("appointment_id")
    if appointment_id is None:
        return {"error": "Missing required parameter: 'appointment_id'."}
    try:
        appointment_id = int(appointment_id)
    except (TypeError, ValueError):
        return {"error": "'appointment_id' must be an integer."}

    appt = dental_db.get_appointment(appointment_id)
    if not appt:
        return {"error": f"Appointment {appointment_id} not found."}
    if appt["status"] == AppointmentStatus.CONFIRMED:
        return {"error": f"Appointment {appointment_id} is already CONFIRMED."}
    if appt["status"] == AppointmentStatus.CANCELLED:
        return {"error": f"Appointment {appointment_id} is already CANCELLED."}

    dental_db.update_appointment_status(appointment_id, AppointmentStatus.CONFIRMED)
    return {
        "status":           AppointmentStatus.CONFIRMED,
        "appointment_id":   appointment_id,
        "patient_name":     appt["patient_name"],
        "date":             str(appt["appointment_date"]),
        "time":             str(appt["appointment_time"]),
        "appointment_type": appt["appointment_type"],
    }


def _handle_cancel(args: dict) -> dict:
    """Cancel an appointment by appointment_id."""
    appointment_id = args.get("appointment_id")
    if appointment_id is None:
        return {"error": "Missing required parameter: 'appointment_id'."}
    try:
        appointment_id = int(appointment_id)
    except (TypeError, ValueError):
        return {"error": "'appointment_id' must be an integer."}

    reason = args.get("reason")
    appt   = dental_db.get_appointment(appointment_id)
    if not appt:
        return {"error": f"Appointment {appointment_id} not found."}
    if appt["status"] == AppointmentStatus.CANCELLED:
        return {"error": f"Appointment {appointment_id} is already CANCELLED."}

    dental_db.update_appointment_status(appointment_id, AppointmentStatus.CANCELLED)
    return {
        "status":           AppointmentStatus.CANCELLED,
        "appointment_id":   appointment_id,
        "patient_name":     appt["patient_name"],
        "date":             str(appt["appointment_date"]),
        "time":             str(appt["appointment_time"]),
        "appointment_type": appt["appointment_type"],
        "reason":           reason,
    }


# ── Tool Factory ──────────────────────────────────────────────────────────────
# Modelled on weather_tool_factory() in function_weather.py

def dental_appointment_tool_factory() -> dict:
    """
    Constructs and returns a JSON-compatible dictionary defining the
    dental_appointment tool for OpenAI LLM function calling.

    Returns
    -------
    dict
        Tool definition dict for openai.chat.completions.create(tools=[...]).
    """
    return {
        "type": "function",
        "function": {
            "name": dental_appointment.__name__,
            "description": (
                "Manage dental clinic appointments for authenticated staff. "
                "Always supply 'action' to select the operation. "
                "Each action requires a specific set of parameters — send only the fields listed for that action:\n"
                "\n"
                "  lookup  — check if a patient already has an active appointment on a given date.\n"
                "            Required: patient_name, date.\n"
                "            Returns: status FOUND (with appointment details) or NOT_FOUND.\n"
                "\n"
                "  book    — schedule a new appointment for a patient.\n"
                "            Required: patient_name, date, time.\n"
                "            Optional: appointment_type (defaults to CHECKUP if omitted).\n"
                "            Returns: status BOOKED (with appointment_id) if the slot is free,\n"
                "                     or status UNAVAILABLE with the next available date/time,\n"
                "                     or status PENDING_APPROVAL if appointment_type is EMERGENCY.\n"
                "            Note: always run lookup first to avoid double-booking.\n"
                "\n"
                "  confirm — mark a SCHEDULED appointment as confirmed.\n"
                "            Required: appointment_id (the integer returned by a prior book call).\n"
                "            Returns: status CONFIRMED with patient and appointment details.\n"
                "\n"
                "  cancel  — cancel an appointment.\n"
                "            Required: appointment_id (the integer returned by a prior book call).\n"
                "            Optional: reason (free-text, recorded in the audit log).\n"
                "            Returns: status CANCELLED with patient and appointment details.\n"
                "\n"
                "On any error the tool returns {\"error\": \"<message>\"} — relay this to the staff member."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": DentalAction.all(),
                        "description": (
                            "The scheduling operation to perform. "
                            "lookup  → patient_name + date. "
                            "book    → patient_name + date + time [+ appointment_type]. "
                            "confirm → appointment_id. "
                            "cancel  → appointment_id [+ reason]."
                        ),
                    },
                    "patient_name": {
                        "type": "string",
                        "description": (
                            "Full name of the patient exactly as it appears in clinic records, "
                            "e.g. 'Jane Smith'. "
                            "Required for: lookup, book. "
                            "Not used for: confirm, cancel."
                        ),
                    },
                    "date": {
                        "type": "string",
                        "description": (
                            "Appointment date in ISO 8601 format: YYYY-MM-DD, e.g. '2026-06-04'. "
                            "Must be today or a future weekday within the next 7 days. "
                            "Required for: lookup, book. "
                            "Not used for: confirm, cancel."
                        ),
                    },
                    "time": {
                        "type": "string",
                        "description": (
                            "Appointment time in 24-hour HH:MM format, e.g. '14:00'. "
                            f"Valid range: {CLINIC_OPEN_HOUR:02d}:00 to {CLINIC_CLOSE_HOUR - 1:02d}:30, weekdays only, in 30-minute slots. "
                            "If the requested slot is taken the tool returns the next available time. "
                            "Required for: book. "
                            "Not used for: lookup, confirm, cancel."
                        ),
                    },
                    "appointment_type": {
                        "type": "string",
                        "enum": AppointmentType.all(),
                        "description": (
                            "Category of dental appointment. "
                            "Supported: CHECKUP, CLEANING, XRAY, CONSULTATION, EMERGENCY. "
                            "EMERGENCY requires supervisor pre-approval and will not be booked immediately. "
                            "Optional for: book — defaults to CHECKUP when omitted. "
                            "Not used for: lookup, confirm, cancel."
                        ),
                    },
                    "appointment_id": {
                        "type": "integer",
                        "description": (
                            "The integer appointment ID returned in the 'appointment_id' field "
                            "of a previous book (status=BOOKED) or lookup (status=FOUND) result. "
                            "Required for: confirm, cancel. "
                            "Not used for: lookup, book."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Free-text reason for the cancellation, stored in the audit log, "
                            "e.g. 'Patient called to reschedule'. "
                            "Optional for: cancel. "
                            "Not used for: lookup, book, confirm."
                        ),
                    },
                },
                "required": ["action"],
            },
        },
    }
