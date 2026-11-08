#!/usr/bin/env python3
"""Local Verda preflight guard; this is not a provider-enforced spending cap.

Run under an independent, awake process supervisor (e.g. macOS caffeinate).
The allocation JSON is private (0600), atomically written after allocation:
  {"schema_version":1,"provider":"verda","instance_id":"ID","hostname":"NAME",
   "requested_at":"2026-10-26T12:00:00Z","initial_balance_usd":100,
   "deadline_seconds":7200,"spend_limit_usd":12,"status":"active"}
Only status may subsequently change; complete/abort requests immediate cleanup.
The script never creates instances or deletes volumes. A missing allocation file
means ready but not yet armed for a target; failed allocation-record publication
must be recovered by the operator. Preserve logs/status until cleanup is verified.
"""
import argparse
import configparser
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import tempfile
import time
import urllib.error
import urllib.request

BASE_URL = "https://api.verda.com/v1"
TERMINAL = {"discontinued", "deleted", "notfound"}
FAILED = {"error", "installation_failed", "no_capacity"}
MAX_RESPONSE_BYTES = 1_048_576


class GuardError(Exception):
    """Messages are deliberately safe to write to logs (no HTTP bodies/secrets)."""


class IdentityError(GuardError):
    pass


class ApiError(GuardError):
    def __init__(self, status=None):
        self.status = status
        super().__init__(f"api_http_{status}" if status else "api_transport_or_response_error")


def private_read(path):
    """Reject symlinks, other owners, and files readable/writable by other users."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "r", encoding="utf-8") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise GuardError("file_requires_current_owner_and_private_regular_mode")
        return stream.read(MAX_RESPONSE_BYTES + 1)


def finite_number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise GuardError(f"invalid_{name}")
    if not low <= value <= high:
        raise GuardError(f"invalid_{name}")
    return float(value)


def load_allocation(path, *, max_deadline_seconds=7200, max_spend_limit_usd=12,
                    max_initial_balance_usd=100):
    try:
        raw = private_read(path)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise GuardError("allocation_too_large")
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("schema_version") != 1 or value.get("provider") != "verda":
            raise GuardError("invalid_allocation_schema")
        for name in ("instance_id", "hostname"):
            if not isinstance(value.get(name), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value[name]):
                raise GuardError(f"invalid_{name}")
        moment = datetime.fromisoformat(value["requested_at"].replace("Z", "+00:00"))
        if moment.tzinfo is None:
            raise GuardError("requested_at_requires_timezone")
        requested_at = moment.timestamp()
        finite_number(requested_at, "requested_at", 1, 10**11)
        finite_number(max_initial_balance_usd, "maximum_initial_balance", 0.01, 450)
        value["initial_balance_usd"] = finite_number(value["initial_balance_usd"], "initial_balance_usd", 0.01, max_initial_balance_usd)
        finite_number(max_deadline_seconds, "maximum_deadline", 1, 345600)
        finite_number(max_spend_limit_usd, "maximum_spend", 0.01, 450)
        value["deadline_seconds"] = finite_number(value["deadline_seconds"], "deadline_seconds", 1, max_deadline_seconds)
        value["spend_limit_usd"] = finite_number(value["spend_limit_usd"], "spend_limit_usd", 0.01, max_spend_limit_usd)
        if value["spend_limit_usd"] >= value["initial_balance_usd"]:
            raise GuardError("spend_limit_requires_prepaid_balance_reserve")
        value["requested_at_epoch"] = requested_at
        value.setdefault("status", "active")
        if value["status"] not in {"active", "complete", "abort"}:
            raise GuardError("invalid_allocation_status")
        return value
    except FileNotFoundError:
        raise
    except GuardError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        raise GuardError("invalid_allocation_file") from None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ApiError(code)


class VerdaAPI:
    def __init__(self, credentials, *, opener=None, clock=time.time, timeout=10):
        self.credentials = Path(credentials).expanduser()
        self.opener = opener or urllib.request.build_opener(NoRedirect())
        self.clock, self.timeout = clock, timeout
        self.token = self.refresh_token = None
        self.expires_at = 0

    def _raw(self, method, path, body=None, token=None):
        headers = {"Accept": "application/json"}
        payload = None
        if body is not None:
            payload = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(BASE_URL + path, data=payload, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                status_code = response.status
                data = response.read(MAX_RESPONSE_BYTES + 1)
            if not 200 <= status_code < 300 or len(data) > MAX_RESPONSE_BYTES:
                raise ApiError(status_code)
            return status_code, json.loads(data) if data else None
        except urllib.error.HTTPError as exc:
            raise ApiError(exc.code) from None
        except (OSError, ValueError, urllib.error.URLError):
            raise ApiError() from None

    def _authenticate(self, force_client=False):
        if self.refresh_token and not force_client:
            try:
                _, result = self._raw("POST", "/oauth2/token", {"grant_type": "refresh_token", "refresh_token": self.refresh_token})
                self._set_token(result)
                return
            except GuardError:
                pass
        try:
            config = configparser.ConfigParser(interpolation=None)
            config.read_string(private_read(self.credentials))
            section = config["default"]
            if section["verda_base_url"].rstrip("/") != BASE_URL:
                raise GuardError("credentials_base_url_must_be_official_verda_api")
            body = {"grant_type": "client_credentials", "client_id": section["verda_client_id"],
                    "client_secret": section["verda_client_secret"]}
            if not body["client_id"] or not body["client_secret"]:
                raise GuardError("invalid_credentials_file")
        except GuardError:
            raise
        except (OSError, KeyError, configparser.Error):
            raise GuardError("invalid_credentials_file") from None
        _, result = self._raw("POST", "/oauth2/token", body)
        self._set_token(result)

    def _set_token(self, result):
        if not isinstance(result, dict) or not isinstance(result.get("access_token"), str) or not result["access_token"]:
            raise GuardError("invalid_auth_response")
        duration = finite_number(result.get("expires_in"), "token_expiry", 1, 10**9)
        self.token = result["access_token"]
        self.refresh_token = result.get("refresh_token")
        self.expires_at = self.clock() + duration

    def request(self, method, path, body=None):
        if not self.token or self.clock() >= self.expires_at - 30:
            self._authenticate()
        try:
            return self._raw(method, path, body, self.token)
        except ApiError as exc:
            if exc.status != 401:
                raise
        self._authenticate(force_client=True)
        return self._raw(method, path, body, self.token)


class Journal:
    def __init__(self, directory, clock=time.time):
        self.directory, self.clock = Path(directory), clock
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise GuardError("state_directory_requires_current_owner_and_mode_0700")
        self.lock = os.open(self.directory / "watchdog.lock", os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.lock)
            raise GuardError("watchdog_already_running_for_state_directory") from None

    def write(self, state, **details):
        value = {"schema_version": 1, "pid": os.getpid(), "updated_at": datetime.fromtimestamp(self.clock(), timezone.utc).isoformat(),
                 "state": state, "ready": True, "armed": state in {"monitoring", "deleting", "monitor_error"}, **details}
        fd = os.open(self.directory / "events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise GuardError("event_log_requires_private_regular_mode")
            stream.write(json.dumps(value, sort_keys=True) + "\n")
            stream.flush()
        with tempfile.NamedTemporaryFile(mode="w", dir=self.directory, prefix=".status-", delete=False) as stream:
            temp = stream.name
            json.dump(value, stream, indent=2)
        os.replace(temp, self.directory / "status.json")

    def close(self):
        os.close(self.lock)


def delete_result_ok(status_code, result, instance_id):
    if status_code == 204:
        return True
    if status_code not in {200, 202, 207} or not isinstance(result, list) or not result:
        return False
    # HTTP 207 can contain errors despite being a 2xx response.
    return all(isinstance(item, dict) and item.get("instanceId") == instance_id
               and item.get("action") == "delete" and item.get("status") == "success" for item in result)


class Watchdog:
    def __init__(self, api, allocation_path, journal, *, clock=time.time, sleep=time.sleep,
                 poll_seconds=20, cleanup_attempts=8, failure_limit=3,
                 max_deadline_seconds=7200, max_spend_limit_usd=12, max_initial_balance_usd=100):
        if not 1 <= poll_seconds <= 30 or not 1 <= cleanup_attempts <= 30 or not 1 <= failure_limit <= 10:
            raise GuardError("invalid_watchdog_limits")
        self.api, self.path, self.journal = api, Path(allocation_path), journal
        self.clock, self.sleep = clock, sleep
        self.poll, self.cleanup_attempts, self.failure_limit = poll_seconds, cleanup_attempts, failure_limit
        self.max_deadline = finite_number(max_deadline_seconds, "maximum_deadline", 1, 345600)
        self.max_spend = finite_number(max_spend_limit_usd, "maximum_spend", 0.01, 450)
        self.max_initial_balance = finite_number(max_initial_balance_usd, "maximum_initial_balance", 0.01, 450)
        self.bound = None
        self.identity_verified = False
        self.stop_requested = False

    def signal_stop(self, *_):
        self.stop_requested = True

    def event(self, state, **details):
        if self.bound:
            details.update(instance_id=self.bound["instance_id"], hostname=self.bound["hostname"],
                           deadline_epoch=self.bound["requested_at_epoch"] + self.bound["deadline_seconds"])
        details["armed"] = bool(self.bound) and state not in {"cleanup_verified", "cleanup_unverified", "identity_mismatch"}
        try:
            self.journal.write(state, **details)
        except (GuardError, OSError) as exc:
            if not self.bound:
                raise
            # Log-storage failures must not stop an already bound guard cleaning up.
            print(json.dumps({"state": "journal_failed", "error": self.safe_error(exc), "action_required": True}), flush=True)

    def _instance(self):
        try:
            _, instance = self.api.request("GET", "/instances/" + self.bound["instance_id"])
        except ApiError as exc:
            if exc.status == 404 and self.identity_verified:
                return "absent"
            raise
        if not isinstance(instance, dict) or instance.get("id") != self.bound["instance_id"] or instance.get("hostname") != self.bound["hostname"]:
            raise IdentityError("instance_identity_mismatch_refusing_delete")
        self.identity_verified = True
        state = instance.get("status")
        if not isinstance(state, str):
            raise GuardError("invalid_instance_status")
        return state

    def cleanup(self, reason):
        """Re-check exact identity immediately before each delete; never delete disks."""
        for attempt in range(1, self.cleanup_attempts + 1):
            self.event("deleting", reason=reason, attempt=attempt)
            try:
                state = self._instance()
                if state == "absent" or state in TERMINAL:
                    self.event("cleanup_verified", reason=reason, instance_status=state)
                    return 0
                if state != "deleting":
                    payload = {"action": "delete", "id": self.bound["instance_id"], "volume_ids": []}
                    code, result = self.api.request("PUT", "/instances", payload)
                    if not delete_result_ok(code, result, self.bound["instance_id"]):
                        raise GuardError("delete_action_response_not_successful")
                    self.event("delete_accepted", reason=reason, attempt=attempt)
                # Acceptance alone is not confirmation that billing/allocation ended.
                state = self._instance()
                if state == "absent" or state in TERMINAL:
                    self.event("cleanup_verified", reason=reason, instance_status=state)
                    return 0
            except IdentityError:
                self.event("identity_mismatch", reason="instance_identity_mismatch_refusing_delete")
                return 2
            except (GuardError, OSError) as exc:
                self.event("cleanup_retry", reason=reason, attempt=attempt, error=self.safe_error(exc))
            if attempt < self.cleanup_attempts:
                self.sleep(self.poll)
        self.event("cleanup_unverified", reason=reason, attempts=self.cleanup_attempts, action_required=True)
        return 3

    @staticmethod
    def safe_error(exc):
        return str(exc) if isinstance(exc, GuardError) else type(exc).__name__

    def run(self):
        failures = 0
        self.event("armed_waiting")
        while True:
            try:
                try:
                    allocation = load_allocation(self.path, max_deadline_seconds=self.max_deadline,
                                                 max_spend_limit_usd=self.max_spend,
                                                 max_initial_balance_usd=self.max_initial_balance)
                except FileNotFoundError:
                    if self.bound:
                        raise GuardError("bound_allocation_file_disappeared")
                    if self.stop_requested:
                        self.event("stopped_without_allocation")
                        return 0
                    self.event("armed_waiting")
                    self.sleep(self.poll)
                    continue
                if self.bound is None:
                    self.bound = allocation
                    if allocation["requested_at_epoch"] > self.clock() + 5:
                        return self.cleanup("allocation_requested_at_is_in_future")
                elif {k: v for k, v in allocation.items() if k != "status"} != {k: v for k, v in self.bound.items() if k != "status"}:
                    # Never accept a new target or an extended deadline mid-run.
                    return self.cleanup("allocation_record_changed")
                if self.stop_requested or allocation["status"] in {"complete", "abort"}:
                    return self.cleanup("stop_requested" if self.stop_requested else allocation["status"])
                if self.clock() >= self.bound["requested_at_epoch"] + self.bound["deadline_seconds"]:
                    return self.cleanup("deadline")
                state = self._instance()
                if state == "absent" or state in TERMINAL:
                    self.event("cleanup_verified", reason="already_terminal", instance_status=state)
                    return 0
                if state in FAILED or state == "deleting":
                    return self.cleanup("instance_" + state)
                _, balance = self.api.request("GET", "/balance")
                if not isinstance(balance, dict) or balance.get("currency", "").lower() != "usd":
                    raise GuardError("balance_must_be_usd")
                remaining = finite_number(balance.get("amount"), "balance", -10**6, 10**9)
                spent = self.bound["initial_balance_usd"] - remaining
                if spent >= self.bound["spend_limit_usd"]:
                    return self.cleanup("spend_limit")
                failures = 0
                self.event("monitoring", instance_status=state, balance_usd=remaining, spend_delta_usd=spent)
            except IdentityError:
                self.event("identity_mismatch", reason="instance_identity_mismatch_refusing_delete", action_required=True)
                return 2
            except (GuardError, OSError) as exc:
                failures += 1
                self.event("monitor_error", consecutive_failures=failures, error=self.safe_error(exc))
                if failures >= self.failure_limit:
                    if self.bound:
                        return self.cleanup("monitor_failure")
                    self.event("invalid_allocation", action_required=True)
                    return 2
            # Do not oversleep the absolute deadline; request timeouts can still delay cleanup.
            delay = self.poll
            if self.bound:
                delay = max(0, min(delay, self.bound["requested_at_epoch"] + self.bound["deadline_seconds"] - self.clock()))
            self.sleep(delay)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, default=Path("~/.config/drivebench/verda/credentials"))
    parser.add_argument("--poll-seconds", type=float, default=20)
    parser.add_argument("--cleanup-attempts", type=int, default=8)
    parser.add_argument("--max-deadline-seconds", type=float, default=7200,
                        help="Explicit ceiling for a NEW allocation; never extends an armed allocation")
    parser.add_argument("--max-spend-limit-usd", type=float, default=12,
                        help="Explicit ceiling for a NEW allocation; prepaid reserve remains required")
    parser.add_argument("--max-initial-balance-usd", type=float, default=100,
                        help="Explicit funded-balance ceiling for a NEW allocation; does not authorize spending")
    args = parser.parse_args(argv)
    journal = None
    try:
        journal = Journal(args.state_dir)
        guard = Watchdog(VerdaAPI(args.credentials), args.allocation, journal, poll_seconds=args.poll_seconds,
                         cleanup_attempts=args.cleanup_attempts, max_deadline_seconds=args.max_deadline_seconds,
                         max_spend_limit_usd=args.max_spend_limit_usd,
                         max_initial_balance_usd=args.max_initial_balance_usd)
        signal.signal(signal.SIGTERM, guard.signal_stop)
        signal.signal(signal.SIGINT, guard.signal_stop)
        return guard.run()
    except (GuardError, OSError) as exc:
        # Do not print exception bodies from urllib, config parsers, or filesystem contents.
        print(json.dumps({"state": "watchdog_failed", "error": Watchdog.safe_error(exc), "action_required": True}), flush=True)
        return 2
    finally:
        if journal:
            journal.close()


if __name__ == "__main__":
    raise SystemExit(main())
