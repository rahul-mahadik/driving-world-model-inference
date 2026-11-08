"""Verda cleanup contract tested entirely with fake HTTP and fake clocks."""
import importlib.util
import io
import json
from pathlib import Path
import urllib.error

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verda_watchdog.py"
spec = importlib.util.spec_from_file_location("verda_watchdog", SCRIPT)
wd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wd)


class Clock:
    now = 1_000_000_000.0

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class Events:
    def __init__(self):
        self.events = []

    def write(self, state, **details):
        self.events.append({"state": state, **details})


class FakeAPI:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def request(self, method, path, body=None):
        self.calls.append((method, path, body))
        assert self.responses, f"Unexpected call {self.calls[-1]}"
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def instance(status="running", **changes):
    return 200, {"id": "instance-test", "hostname": "drivebench-preflight", "status": status, **changes}


def accepted(status="success", code=202):
    return code, [{"instanceId": "instance-test", "action": "delete", "status": status}]


def record(tmp_path, clock, **changes):
    value = {"schema_version": 1, "provider": "verda", "instance_id": "instance-test",
             "hostname": "drivebench-preflight", "requested_at": wd.datetime.fromtimestamp(clock(), wd.timezone.utc).isoformat(),
             "initial_balance_usd": 100, "deadline_seconds": 7200, "spend_limit_usd": 12, "status": "active", **changes}
    path = tmp_path / "allocation.json"
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    return path


def guard(tmp_path, responses, **changes):
    clock, events = Clock(), Events()
    path = record(tmp_path, clock, **changes)
    api = FakeAPI(responses)
    obj = wd.Watchdog(api, path, events, clock=clock, sleep=clock.sleep, poll_seconds=20, cleanup_attempts=3)
    return obj, api, clock, events


def puts(api):
    return [call for call in api.calls if call[0] == "PUT"]


def test_absolute_deadline_deletes_preserving_every_volume(tmp_path):
    obj, api, clock, events = guard(tmp_path, [instance(), (200, {"amount": 100, "currency": "usd"}),
                                             instance(), accepted(), wd.ApiError(404)], deadline_seconds=1)
    assert obj.run() == 0
    assert clock.now == 1_000_000_001
    assert puts(api) == [("PUT", "/instances", {"action": "delete", "id": "instance-test", "volume_ids": []})]
    assert events.events[-1]["state"] == "cleanup_verified"
    assert events.events[-1]["reason"] == "deadline"


@pytest.mark.parametrize("amount,reason", [(88, "spend_limit"), (87.99, "spend_limit")])
def test_spend_boundary(tmp_path, amount, reason):
    obj, api, _, events = guard(tmp_path, [instance(), (200, {"amount": amount, "currency": "USD"}),
                                         instance(), accepted(), instance("discontinued")])
    assert obj.run() == 0
    assert len(puts(api)) == 1
    assert events.events[-1]["reason"] == reason


@pytest.mark.parametrize("change", [{"id": "someone-else"}, {"hostname": "someone-else"}])
def test_identity_mismatch_never_deletes(tmp_path, change):
    obj, api, _, events = guard(tmp_path, [instance(**change)], status="complete")
    assert obj.run() == 2
    assert puts(api) == []
    assert events.events[-1]["state"] == "identity_mismatch"


def test_recheck_identity_before_delete_after_monitoring(tmp_path):
    obj, api, _, _ = guard(tmp_path, [instance(), (200, {"amount": 88, "currency": "usd"}), instance(hostname="changed")])
    assert obj.run() == 2
    assert puts(api) == []


@pytest.mark.parametrize("status", ["discontinued", "deleted", "notfound"])
def test_already_terminal_needs_no_mutation(tmp_path, status):
    obj, api, _, events = guard(tmp_path, [instance(status)])
    assert obj.run() == 0
    assert puts(api) == []
    assert events.events[-1]["instance_status"] == status


@pytest.mark.parametrize("status", ["complete", "abort"])
def test_completion_or_abort_requests_cleanup(tmp_path, status):
    obj, api, _, events = guard(tmp_path, [instance("offline"), accepted(), instance("discontinued")], status=status)
    assert obj.run() == 0
    assert len(puts(api)) == 1  # offline/shutdown must still be deleted
    assert events.events[-1]["reason"] == status


def test_http207_error_is_not_accepted_and_is_retried(tmp_path):
    obj, api, clock, events = guard(tmp_path, [instance(), accepted("error", 207), instance(), accepted(), wd.ApiError(404)], status="abort")
    assert obj.run() == 0
    assert len(puts(api)) == 2
    assert clock.now == 1_000_000_020
    assert any(e["state"] == "cleanup_retry" for e in events.events)


def test_success_response_without_terminal_status_is_not_cleanup_proof(tmp_path):
    responses = []
    for _ in range(3):
        responses += [instance(), accepted(), instance("deleting")]
    obj, api, _, events = guard(tmp_path, responses, status="complete")
    assert obj.run() == 3
    assert events.events[-1]["state"] == "cleanup_unverified"
    assert events.events[-1]["action_required"]
    assert len(puts(api)) == 3


def test_get_failure_does_not_allow_unverified_delete_and_retries_are_bounded(tmp_path):
    obj, api, _, events = guard(tmp_path, [wd.ApiError(503)] * 3, status="abort")
    assert obj.run() == 3
    assert puts(api) == []
    assert len(api.calls) == 3
    assert events.events[-1]["state"] == "cleanup_unverified"


def test_initial404_is_not_treated_as_deleted_before_identity_seen(tmp_path):
    obj, api, _, events = guard(tmp_path, [wd.ApiError(404), instance(), accepted(), wd.ApiError(404)], status="complete")
    assert obj.run() == 0
    assert len(puts(api)) == 1
    assert any(e["state"] == "cleanup_retry" for e in events.events)


def test_monitor_failure_policy_deletes_after_three_errors(tmp_path):
    responses = []
    for _ in range(3):
        responses += [instance(), wd.ApiError(503)]
    obj, api, _, events = guard(tmp_path, responses + [instance(), accepted(), instance("discontinued")])
    assert obj.run() == 0
    assert events.events[-1]["reason"] == "monitor_failure"
    assert len(puts(api)) == 1


def test_wait_before_allocation_does_not_read_credentials_or_call_api(tmp_path):
    clock, events = Clock(), Events()
    api = FakeAPI([instance("discontinued")])
    path = tmp_path / "allocation.json"
    def sleep(duration):
        assert not api.calls
        assert events.events[-1]["state"] == "armed_waiting"
        assert not events.events[-1]["armed"]
        clock.sleep(duration)
        record(tmp_path, clock)
    obj = wd.Watchdog(api, path, events, clock=clock, sleep=sleep)
    assert obj.run() == 0
    assert len(api.calls) == 1


@pytest.mark.parametrize("changes", [{"deadline_seconds": 7201}, {"spend_limit_usd": 13},
                                      {"requested_at": "2026-01-01"}, {"initial_balance_usd": 101},
                                      {"instance_id": "../../somewhere"}, {"spend_limit_usd": float("nan")}])
def test_reject_unsafe_allocation_fields(tmp_path, changes):
    with pytest.raises(wd.GuardError):
        wd.load_allocation(record(tmp_path, Clock(), **changes))


def test_reject_world_readable_allocation_and_symlink(tmp_path):
    path = record(tmp_path, Clock())
    path.chmod(0o644)
    with pytest.raises(wd.GuardError):
        wd.load_allocation(path)
    path.chmod(0o600)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(wd.GuardError):
        wd.load_allocation(link)


def test_record_retargeting_cleans_up_original_id_only(tmp_path):
    obj, api, clock, events = guard(tmp_path, [instance(), (200, {"amount": 100, "currency": "usd"}),
                                             instance(), accepted(), instance("discontinued")])
    def sleep(duration):
        clock.sleep(duration)
        record(tmp_path, clock, instance_id="other-instance")
    obj.sleep = sleep
    assert obj.run() == 0
    assert puts(api)[0][2]["id"] == "instance-test"
    assert events.events[-1]["reason"] == "allocation_record_changed"


class Response(io.BytesIO):
    def __init__(self, body, status=200):
        super().__init__(json.dumps(body).encode() if body is not None else b"")
        self.status = status


class Opener:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def open(self, request, timeout):
        self.calls.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def credential_file(tmp_path):
    path = tmp_path / "credentials"
    path.write_text("[default]\nverda_base_url=https://api.verda.com/v1\nverda_client_id=FAKE_CLIENT\nverda_client_secret=FAKE_SECRET\n")
    path.chmod(0o600)
    return path


def token(access="FAKE_TOKEN"):
    return Response({"access_token": access, "expires_in": 60, "refresh_token": "FAKE_REFRESH"})


def test_real_http_wrapper_refreshes_expired_token_with_exact_official_payload(tmp_path):
    clock = Clock()
    opener = Opener([token(), Response({"amount": 100}), token("NEW_TOKEN"), Response({"amount": 99})])
    api = wd.VerdaAPI(credential_file(tmp_path), opener=opener, clock=clock)
    api.request("GET", "/balance")
    clock.sleep(60)
    api.request("GET", "/balance")
    assert json.loads(opener.calls[0].data) == {"grant_type": "client_credentials", "client_id": "FAKE_CLIENT", "client_secret": "FAKE_SECRET"}
    assert json.loads(opener.calls[2].data) == {"grant_type": "refresh_token", "refresh_token": "FAKE_REFRESH"}
    assert opener.calls[3].headers["Authorization"] == "Bearer NEW_TOKEN"
    assert all(r.full_url.startswith(wd.BASE_URL + "/") for r in opener.calls)


def test_http401_reauthenticates_and_retries_once(tmp_path):
    denied = urllib.error.HTTPError("https://api.verda.com/v1/balance", 401, "FAKE_SECRET", {}, None)
    opener = Opener([token(), denied, token("NEW_TOKEN"), Response({"amount": 100})])
    api = wd.VerdaAPI(credential_file(tmp_path), opener=opener, clock=Clock())
    assert api.request("GET", "/balance")[1]["amount"] == 100
    assert json.loads(opener.calls[2].data)["grant_type"] == "client_credentials"


def test_refresh_failure_falls_back_to_credentials(tmp_path):
    denied = urllib.error.HTTPError("https://api.verda.com/v1/oauth2/token", 401, "FAKE_SECRET", {}, None)
    clock = Clock()
    opener = Opener([token(), Response({}), denied, token(), Response({})])
    api = wd.VerdaAPI(credential_file(tmp_path), opener=opener, clock=clock)
    api.request("GET", "/balance")
    clock.sleep(60)
    api.request("GET", "/balance")
    assert json.loads(opener.calls[3].data)["grant_type"] == "client_credentials"


def test_transport_errors_never_include_secret_response_body(tmp_path):
    denied = urllib.error.HTTPError("https://api.verda.com/v1/oauth2/token", 403, "FAKE_SECRET", {}, None)
    api = wd.VerdaAPI(credential_file(tmp_path), opener=Opener([denied]), clock=Clock())
    with pytest.raises(wd.ApiError) as error:
        api.request("GET", "/balance")
    assert str(error.value) == "api_http_403"


def test_journal_records_private_heartbeat_and_exclusive_process_lock(tmp_path):
    journal = wd.Journal(tmp_path / "private", Clock())
    try:
        journal.write("armed_waiting", armed=False)
        state = json.loads((journal.directory / "status.json").read_text())
        assert state["ready"] and not state["armed"] and state["updated_at"]
        assert (journal.directory / "status.json").stat().st_mode & 0o077 == 0
        assert (journal.directory / "events.jsonl").stat().st_mode & 0o077 == 0
        with pytest.raises(wd.GuardError, match="already_running"):
            wd.Journal(journal.directory, Clock())
    finally:
        journal.close()


def test_elapsed_deadline_on_restart_deletes_without_resetting_clock(tmp_path):
    obj, api, clock, events = guard(tmp_path, [instance(), accepted(), instance("discontinued")])
    clock.sleep(7300)
    assert obj.run() == 0
    assert events.events[-1]["reason"] == "deadline"
    assert not any(call[1] == "/balance" for call in api.calls)


def test_future_timestamp_cannot_extend_deadline(tmp_path):
    obj, api, _, events = guard(tmp_path, [instance(), accepted(), instance("discontinued")],
                                requested_at="2090-01-01T00:00:00+00:00")
    assert obj.run() == 0
    assert events.events[-1]["reason"] == "allocation_requested_at_is_in_future"


def test_sigterm_requests_preserving_cleanup(tmp_path):
    obj, api, _, events = guard(tmp_path, [instance(), accepted(), instance("discontinued")])
    obj.signal_stop()
    assert obj.run() == 0
    assert events.events[-1]["reason"] == "stop_requested"
    assert puts(api)[0][2]["volume_ids"] == []


def test_deleting_status_waits_for_terminal_without_duplicate_mutation(tmp_path):
    obj, api, _, events = guard(tmp_path, [instance("deleting"), instance("deleting"), instance("deleting"),
                                         instance("discontinued")])
    assert obj.run() == 0
    assert puts(api) == []
    assert events.events[-1]["state"] == "cleanup_verified"


def test_failed_instance_triggers_cleanup(tmp_path):
    obj, api, _, events = guard(tmp_path, [instance("installation_failed"), instance("installation_failed"),
                                         accepted(), instance("discontinued")])
    assert obj.run() == 0
    assert events.events[-1]["reason"] == "instance_installation_failed"


def test_logging_failure_after_binding_does_not_stop_cleanup(tmp_path, capsys):
    obj, api, _, events = guard(tmp_path, [instance(), accepted(), instance("discontinued")], status="abort")
    original_write = events.write
    def write(state, **details):
        if state != "armed_waiting":
            raise OSError("disk failure")
        original_write(state, **details)
    events.write = write
    assert obj.run() == 0
    assert len(puts(api)) == 1
    assert "journal_failed" in capsys.readouterr().out


def test_wrong_credentials_destination_is_rejected_before_any_http(tmp_path):
    path = credential_file(tmp_path)
    path.write_text(path.read_text().replace("https://api.verda.com/v1", "https://untrusted.invalid"))
    opener = Opener([])
    api = wd.VerdaAPI(path, opener=opener, clock=Clock())
    with pytest.raises(wd.GuardError, match="official_verda_api"):
        api.request("GET", "/balance")
    assert opener.calls == []


def test_long_study_requires_explicit_launch_policy(tmp_path):
    path = record(tmp_path, Clock(), deadline_seconds=57600, spend_limit_usd=75, initial_balance_usd=90)
    with pytest.raises(wd.GuardError):
        wd.load_allocation(path)
    result = wd.load_allocation(path, max_deadline_seconds=57600, max_spend_limit_usd=75)
    assert result['deadline_seconds'] == 57600
    assert result['spend_limit_usd'] == 75


def test_long_study_policy_still_deletes_on_spending_boundary(tmp_path):
    clock, events = Clock(), Events()
    path = record(tmp_path, clock, deadline_seconds=57600, spend_limit_usd=75, initial_balance_usd=90)
    api = FakeAPI([instance(), (200, {'amount': 15, 'currency': 'usd'}), instance(), accepted(), instance('discontinued')])
    obj = wd.Watchdog(api, path, events, clock=clock, sleep=clock.sleep,
                     max_deadline_seconds=57600, max_spend_limit_usd=75)
    assert obj.run() == 0
    assert events.events[-1]['reason'] == 'spend_limit'
    assert puts(api)[0][2]['volume_ids'] == []


def test_explicit_study_policy_requires_balance_reserve(tmp_path):
    path = record(tmp_path, Clock(), spend_limit_usd=90, initial_balance_usd=90)
    with pytest.raises(wd.GuardError, match='prepaid_balance_reserve'):
        wd.load_allocation(path, max_spend_limit_usd=90)


@pytest.mark.parametrize('kwargs', [dict(max_deadline_seconds=345601), dict(max_spend_limit_usd=451),
                                     dict(max_deadline_seconds=float('nan'))])
def test_reject_unbounded_study_policy(tmp_path, kwargs):
    with pytest.raises(wd.GuardError):
        wd.Watchdog(FakeAPI([]), tmp_path / 'allocation.json', Events(), **kwargs)


def test_higher_prepaid_balance_requires_explicit_new_policy(tmp_path):
    path = record(tmp_path, Clock(), initial_balance_usd=220)
    with pytest.raises(wd.GuardError, match='initial_balance'):
        wd.load_allocation(path)
    allocation = wd.load_allocation(path, max_initial_balance_usd=300)
    assert allocation['initial_balance_usd'] == 220
    assert allocation['spend_limit_usd'] == 12


def test_funded_allocation_still_stops_at_smaller_spend_limit(tmp_path):
    clock, events = Clock(), Events()
    path = record(tmp_path, clock, initial_balance_usd=220)
    api = FakeAPI([instance(), (200, {'amount': 208, 'currency': 'usd'}), instance(), accepted(), instance('discontinued')])
    obj = wd.Watchdog(api, path, events, clock=clock, sleep=clock.sleep, max_initial_balance_usd=300)
    assert obj.run() == 0
    assert events.events[-1]['reason'] == 'spend_limit'


@pytest.mark.parametrize('value', [451, float('nan'), -1, True])
def test_unbounded_initial_balance_policy_is_rejected(tmp_path, value):
    with pytest.raises(wd.GuardError, match='maximum_initial_balance'):
        wd.Watchdog(FakeAPI([]), tmp_path / 'allocation.json', Events(), max_initial_balance_usd=value)
