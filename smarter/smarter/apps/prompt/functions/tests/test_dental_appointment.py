"""
Tests for function_dental_appointment.py

Coverage
--------
- dental_appointment_tool_factory() — schema shape
- _require_staff_user()             — auth guard (staff / non-staff / no request)
- _validate_date() / _validate_time() — input validation edge cases
- _handle_lookup()                  — FOUND / NOT_FOUND
- _handle_book()                    — BOOKED / UNAVAILABLE / EMERGENCY / duplicate
- _handle_confirm()                 — CONFIRMED / already-confirmed / already-cancelled
- _handle_cancel()                  — CANCELLED / already-cancelled / with reason
- _pick_doctor paths                — preferred hit / preferred miss / fallback / no doctor
"""

import json
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from smarter.lib.unittest.base_classes import SmarterTestBase

from ..function_dental_appointment import (
    _handle_book,
    _handle_cancel,
    _handle_confirm,
    _handle_lookup,
    _require_staff_user,
    _validate_date,
    _validate_time,
    dental_appointment,
    dental_appointment_tool_factory,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

FUTURE_DATE = (date.today() + timedelta(days=7)).strftime("%Y-%m-%d")
PAST_DATE   = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
VALID_TIME  = "10:00"


def _make_tool_call(action: str, extra: dict | None = None) -> ChatCompletionMessageToolCall:
    args = {"action": action}
    if extra:
        args.update(extra)
    return ChatCompletionMessageToolCall(
        id=f"tc_{action}",
        function=Function(name="dental_appointment", arguments=json.dumps(args)),
        type="function",
    )


def _staff_request(is_staff=True, is_authenticated=True):
    user = MagicMock()
    user.is_authenticated = is_authenticated
    user.is_staff = is_staff
    user.username = "staff_user" if is_staff else "regular_user"
    request = MagicMock()
    request.user = user
    return request


# ── Tool Factory ──────────────────────────────────────────────────────────────

class TestDentalAppointmentToolFactory(SmarterTestBase):
    """dental_appointment_tool_factory() returns a valid OpenAI tool definition."""

    def test_returns_dict(self):
        result = dental_appointment_tool_factory()
        self.assertIsInstance(result, dict)

    def test_type_is_function(self):
        result = dental_appointment_tool_factory()
        self.assertEqual(result["type"], "function")

    def test_has_function_key(self):
        result = dental_appointment_tool_factory()
        self.assertIn("function", result)

    def test_function_name(self):
        result = dental_appointment_tool_factory()
        self.assertEqual(result["function"]["name"], "dental_appointment")

    def test_has_action_enum(self):
        result = dental_appointment_tool_factory()
        props = result["function"]["parameters"]["properties"]
        self.assertIn("action", props)
        self.assertIn("enum", props["action"])
        self.assertIn("lookup", props["action"]["enum"])
        self.assertIn("book",   props["action"]["enum"])
        self.assertIn("confirm", props["action"]["enum"])
        self.assertIn("cancel",  props["action"]["enum"])


# ── Auth Guard ────────────────────────────────────────────────────────────────

class TestRequireStaffUser(SmarterTestBase):
    """_require_staff_user() returns None for staff, error string for everyone else."""

    def test_staff_user_allowed(self):
        with patch("smarter.apps.prompt.functions.function_dental_appointment.is_authenticated_request", return_value=True):
            self.assertIsNone(_require_staff_user(_staff_request(is_staff=True)))

    def test_non_staff_user_denied(self):
        with patch("smarter.apps.prompt.functions.function_dental_appointment.is_authenticated_request", return_value=True):
            result = _require_staff_user(_staff_request(is_staff=False))
        self.assertIsNotNone(result)
        self.assertIn("restricted", result)

    def test_no_request_denied(self):
        result = _require_staff_user(None)
        self.assertIsNotNone(result)
        self.assertIn("restricted", result)

    def test_unauthenticated_denied(self):
        with patch("smarter.apps.prompt.functions.function_dental_appointment.is_authenticated_request", return_value=False):
            result = _require_staff_user(_staff_request(is_authenticated=False))
        self.assertIsNotNone(result)


# ── Validation ────────────────────────────────────────────────────────────────

class TestValidateDate(SmarterTestBase):
    """_validate_date() edge cases."""

    def test_valid_future_date(self):
        self.assertIsNone(_validate_date(FUTURE_DATE))

    def test_past_date_rejected(self):
        err = _validate_date(PAST_DATE)
        self.assertIsNotNone(err)
        self.assertIn("past", err)

    def test_bad_format_rejected(self):
        err = _validate_date("13/06/2026")
        self.assertIsNotNone(err)

    def test_invalid_calendar_date(self):
        err = _validate_date("2026-02-30")
        self.assertIsNotNone(err)


class TestValidateTime(SmarterTestBase):
    """_validate_time() slot boundary checks."""

    def test_valid_on_hour(self):
        self.assertIsNone(_validate_time("09:00"))

    def test_valid_on_half_hour(self):
        self.assertIsNone(_validate_time("14:30"))

    def test_last_valid_slot(self):
        self.assertIsNone(_validate_time("16:30"))

    def test_after_close_rejected(self):
        err = _validate_time("17:00")
        self.assertIsNotNone(err)

    def test_before_open_rejected(self):
        err = _validate_time("07:30")
        self.assertIsNotNone(err)

    def test_non_half_hour_rejected(self):
        err = _validate_time("10:15")
        self.assertIsNotNone(err)

    def test_bad_format_rejected(self):
        err = _validate_time("10am")
        self.assertIsNotNone(err)


# ── Lookup Handler ────────────────────────────────────────────────────────────

class TestHandleLookup(SmarterTestBase):
    """_handle_lookup() FOUND and NOT_FOUND paths."""

    def _patch_db(self, row):
        mock_db = MagicMock()
        mock_db.lookup_patient_appointment.return_value = row
        return mock_db

    def test_found_with_doctor(self):
        row = {
            "appointment_id": 42,
            "appointment_time": "10:00:00",
            "appointment_type": "CHECKUP",
            "status": "SCHEDULED",
            "doctor_name": "Doctor Wong",
        }
        with patch("smarter.apps.prompt.functions.function_dental_appointment.dental_db", self._patch_db(row)):
            result = _handle_lookup({"patient_name": "Jane Smith", "date": FUTURE_DATE})
        self.assertEqual(result["status"], "FOUND")
        self.assertEqual(result["appointment_id"], 42)
        self.assertEqual(result["doctor"], "Doctor Wong")

    def test_found_without_doctor(self):
        row = {
            "appointment_id": 7,
            "appointment_time": "09:00:00",
            "appointment_type": "CLEANING",
            "status": "SCHEDULED",
            "doctor_name": None,
        }
        with patch("smarter.apps.prompt.functions.function_dental_appointment.dental_db", self._patch_db(row)):
            result = _handle_lookup({"patient_name": "Bob Lee", "date": FUTURE_DATE})
        self.assertEqual(result["status"], "FOUND")
        self.assertNotIn("doctor", result)

    def test_not_found(self):
        with patch("smarter.apps.prompt.functions.function_dental_appointment.dental_db", self._patch_db(None)):
            result = _handle_lookup({"patient_name": "Jane Smith", "date": FUTURE_DATE})
        self.assertEqual(result["status"], "NOT_FOUND")

    def test_missing_patient_name(self):
        result = _handle_lookup({"date": FUTURE_DATE})
        self.assertIn("error", result)
        self.assertIn("patient_name", result["error"])

    def test_missing_date(self):
        result = _handle_lookup({"patient_name": "Jane Smith"})
        self.assertIn("error", result)

    def test_past_date_rejected(self):
        result = _handle_lookup({"patient_name": "Jane Smith", "date": PAST_DATE})
        self.assertIn("error", result)


# ── Book Handler ──────────────────────────────────────────────────────────────

class TestHandleBook(SmarterTestBase):
    """_handle_book() BOOKED / UNAVAILABLE / EMERGENCY / duplicate paths."""

    def _mock_db(self, *, slot_available=True, existing=None, patient_id=1,
                 appt_id=99, preferred_doc=None, fallback_doc=None, next_slot=None):
        db = MagicMock()
        db.lookup_patient_appointment.return_value = existing
        db.is_slot_available.return_value = slot_available
        db.get_or_create_patient.return_value = patient_id
        db.book_appointment.return_value = appt_id
        db.resolve_doctor.return_value = preferred_doc
        db.is_doctor_available.return_value = preferred_doc is not None
        db.get_any_available_doctor.return_value = fallback_doc
        db.find_next_available.return_value = next_slot
        return db

    def test_booked_with_preferred_doctor(self):
        doc = {"doctor_id": 3, "doctor_name": "Doctor Kim"}
        db = self._mock_db(preferred_doc=doc)
        args = {"patient_name": "Alice", "date": FUTURE_DATE, "time": VALID_TIME,
                "preferred_doctors": ["Doctor Kim"]}
        with patch("smarter.apps.prompt.functions.function_dental_appointment.dental_db", db):
            result = _handle_book(args)
        self.assertEqual(result["status"], "BOOKED")
        self.assertEqual(result["doctor"], "Doctor Kim")
        self.assertEqual(result["appointment_id"], 99)

    def test_booked_fallback_doctor_when_preferred_not_found(self):
        fallback = {"doctor_id": 5, "doctor_name": "Doctor Chen"}
        db = self._mock_db(preferred_doc=None, fallback_doc=fallback)
        args = {"patient_name": "Alice", "date": FUTURE_DATE, "time": VALID_TIME,
                "preferred_doctors": ["Doctor Unknown"]}
        with patch("smarter.apps.prompt.functions.function_dental_appointment.dental_db", db):
            result = _handle_book(args)
        self.assertEqual(result["status"], "BOOKED")
        self.assertEqual(result["doctor"], "Doctor Chen")

    def test_booked_no_doctor_available(self):
        db = self._mock_db(preferred_doc=None, fallback_doc=None)
        args = {"patient_name": "Alice", "date": FUTURE_DATE, "time": VALID_TIME}
        with patch("smarter.apps.prompt.functions.function_dental_appointment.dental_db", db):
            result = _handle_book(args)
        self.assertEqual(result["status"], "BOOKED")
        self.assertNotIn("doctor", result)

    def test_unavailable_with_next_slot(self):
        next_slot = {"date": FUTURE_DATE, "time": "11:00"}
        fallback = {"doctor_id": 2, "doctor_name": "Doctor Patel"}
        db = self._mock_db(slot_available=False, fallback_doc=fallback, next_slot=next_slot)
        args = {"patient_name": "Alice", "date": FUTURE_DATE, "time": VALID_TIME}
        with patch("smarter.apps.prompt.functions.function_dental_appointment.dental_db", db):
            result = _handle_book(args)
        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertIn("next_available_date", result)

    def test_unavailable_no_next_slot(self):
        db = self._mock_db(slot_available=False, next_slot=None)
        args = {"patient_name": "Alice", "date": FUTURE_DATE, "time": VALID_TIME}
        with patch("smarter.apps.prompt.functions.function_dental_appointment.dental_db", db):
            result = _handle_book(args)
        self.assertIn("error", result)

    def test_emergency_returns_pending_approval(self):
        db = self._mock_db()
        args = {"patient_name": "Alice", "date": FUTURE_DATE, "time": VALID_TIME,
                "appointment_type": "EMERGENCY"}
        with patch("smarter.apps.prompt.functions.function_dental_appointment.dental_db", db):
            result = _handle_book(args)
        self.assertEqual(result["status"], "PENDING_APPROVAL")
        self.assertNotIn("appointment_id", result)
        self.assertNotIn("doctor", result)

    def test_duplicate_booking_rejected(self):
        existing = {"appointment_type": "CHECKUP", "appointment_time": "09:00:00"}
        db = self._mock_db(existing=existing)
        args = {"patient_name": "Alice", "date": FUTURE_DATE, "time": VALID_TIME}
        with patch("smarter.apps.prompt.functions.function_dental_appointment.dental_db", db):
            result = _handle_book(args)
        self.assertIn("error", result)
        self.assertIn("already has", result["error"])

    def test_missing_patient_name(self):
        result = _handle_book({"date": FUTURE_DATE, "time": VALID_TIME})
        self.assertIn("error", result)

    def test_invalid_appointment_type(self):
        result = _handle_book({"patient_name": "Alice", "date": FUTURE_DATE,
                                "time": VALID_TIME, "appointment_type": "MASSAGE"})
        self.assertIn("error", result)


# ── Confirm Handler ───────────────────────────────────────────────────────────

class TestHandleConfirm(SmarterTestBase):
    """_handle_confirm() CONFIRMED and error paths."""

    def _mock_db(self, appt):
        db = MagicMock()
        db.get_appointment.return_value = appt
        return db

    def _scheduled_appt(self):
        return {
            "appointment_id": 10, "patient_name": "Bob",
            "appointment_date": FUTURE_DATE, "appointment_time": "10:00:00",
            "appointment_type": "CHECKUP", "status": "SCHEDULED",
            "doctor_name": "Doctor Wong",
        }

    def test_confirm_scheduled(self):
        db = self._mock_db(self._scheduled_appt())
        with patch("smarter.apps.prompt.functions.function_dental_appointment.dental_db", db):
            result = _handle_confirm({"appointment_id": 10})
        self.assertEqual(result["status"], "CONFIRMED")
        db.update_appointment_status.assert_called_once_with(10, "CONFIRMED")

    def test_already_confirmed_error(self):
        appt = self._scheduled_appt()
        appt["status"] = "CONFIRMED"
        db = self._mock_db(appt)
        with patch("smarter.apps.prompt.functions.function_dental_appointment.dental_db", db):
            result = _handle_confirm({"appointment_id": 10})
        self.assertIn("error", result)
        self.assertIn("already CONFIRMED", result["error"])

    def test_already_cancelled_error(self):
        appt = self._scheduled_appt()
        appt["status"] = "CANCELLED"
        db = self._mock_db(appt)
        with patch("smarter.apps.prompt.functions.function_dental_appointment.dental_db", db):
            result = _handle_confirm({"appointment_id": 10})
        self.assertIn("error", result)

    def test_not_found_error(self):
        db = self._mock_db(None)
        with patch("smarter.apps.prompt.functions.function_dental_appointment.dental_db", db):
            result = _handle_confirm({"appointment_id": 999})
        self.assertIn("error", result)

    def test_missing_appointment_id(self):
        result = _handle_confirm({})
        self.assertIn("error", result)
        self.assertIn("appointment_id", result["error"])

    def test_non_integer_appointment_id(self):
        result = _handle_confirm({"appointment_id": "abc"})
        self.assertIn("error", result)


# ── Cancel Handler ────────────────────────────────────────────────────────────

class TestHandleCancel(SmarterTestBase):
    """_handle_cancel() CANCELLED and error paths."""

    def _mock_db(self, appt):
        db = MagicMock()
        db.get_appointment.return_value = appt
        return db

    def _scheduled_appt(self):
        return {
            "appointment_id": 20, "patient_name": "Carol",
            "appointment_date": FUTURE_DATE, "appointment_time": "14:00:00",
            "appointment_type": "XRAY", "status": "SCHEDULED",
            "doctor_name": None,
        }

    def test_cancel_scheduled(self):
        db = self._mock_db(self._scheduled_appt())
        with patch("smarter.apps.prompt.functions.function_dental_appointment.dental_db", db):
            result = _handle_cancel({"appointment_id": 20})
        self.assertEqual(result["status"], "CANCELLED")
        db.update_appointment_status.assert_called_once_with(20, "CANCELLED")

    def test_cancel_with_reason(self):
        db = self._mock_db(self._scheduled_appt())
        with patch("smarter.apps.prompt.functions.function_dental_appointment.dental_db", db):
            result = _handle_cancel({"appointment_id": 20, "reason": "Patient request"})
        self.assertEqual(result["reason"], "Patient request")

    def test_already_cancelled_error(self):
        appt = self._scheduled_appt()
        appt["status"] = "CANCELLED"
        db = self._mock_db(appt)
        with patch("smarter.apps.prompt.functions.function_dental_appointment.dental_db", db):
            result = _handle_cancel({"appointment_id": 20})
        self.assertIn("error", result)

    def test_not_found_error(self):
        db = self._mock_db(None)
        with patch("smarter.apps.prompt.functions.function_dental_appointment.dental_db", db):
            result = _handle_cancel({"appointment_id": 999})
        self.assertIn("error", result)

    def test_missing_appointment_id(self):
        result = _handle_cancel({})
        self.assertIn("error", result)


# ── End-to-end dental_appointment() ──────────────────────────────────────────

class TestDentalAppointmentEntryPoint(SmarterTestBase):
    """dental_appointment() auth gate and dispatch via mocked DB."""

    def _mock_db_lookup_found(self):
        db = MagicMock()
        db.lookup_patient_appointment.return_value = {
            "appointment_id": 1, "appointment_time": "10:00:00",
            "appointment_type": "CHECKUP", "status": "SCHEDULED",
            "doctor_name": "Doctor Wong",
        }
        return db

    def test_auth_rejected_no_request(self):
        tool_call = _make_tool_call("lookup", {"patient_name": "Jane", "date": FUTURE_DATE})
        result = dental_appointment(tool_call, request=None)
        self.assertIsInstance(result, list)
        self.assertIn("error", result[0])
        self.assertIn("restricted", result[0]["error"])

    def test_auth_rejected_non_staff(self):
        tool_call = _make_tool_call("lookup", {"patient_name": "Jane", "date": FUTURE_DATE})
        with patch("smarter.apps.prompt.functions.function_dental_appointment.is_authenticated_request", return_value=True):
            result = dental_appointment(tool_call, request=_staff_request(is_staff=False))
        self.assertIn("error", result[0])

    def test_dispatches_lookup_for_staff(self):
        tool_call = _make_tool_call("lookup", {"patient_name": "Jane", "date": FUTURE_DATE})
        db = self._mock_db_lookup_found()
        with patch("smarter.apps.prompt.functions.function_dental_appointment.is_authenticated_request", return_value=True), \
             patch("smarter.apps.prompt.functions.function_dental_appointment.dental_db", db), \
             patch("smarter.apps.prompt.functions.function_dental_appointment.llm_tool_requested"), \
             patch("smarter.apps.prompt.functions.function_dental_appointment.llm_tool_responded"):
            result = dental_appointment(tool_call, request=_staff_request())
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["status"], "FOUND")

    def test_invalid_action_returns_error(self):
        tool_call = _make_tool_call("teleport")
        with patch("smarter.apps.prompt.functions.function_dental_appointment.is_authenticated_request", return_value=True), \
             patch("smarter.apps.prompt.functions.function_dental_appointment.llm_tool_requested"):
            result = dental_appointment(tool_call, request=_staff_request())
        self.assertIn("error", result[0])

    def test_malformed_json_arguments(self):
        function = Function(name="dental_appointment", arguments="{invalid json")
        tool_call = ChatCompletionMessageToolCall(id="tc_bad", function=function, type="function")
        with patch("smarter.apps.prompt.functions.function_dental_appointment.is_authenticated_request", return_value=True):
            result = dental_appointment(tool_call, request=_staff_request())
        self.assertIn("error", result[0])
