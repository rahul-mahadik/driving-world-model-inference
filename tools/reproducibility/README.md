# Installed-package reproduction check

Run `check_install.py` with the Python interpreter containing non-editable wheel
installations of both packages. The output must be a new directory outside the
source checkout. Full commands and GPU reproduction boundaries are in
[the release guide](../../docs/reproduce-release.md).

The checker calls the existing synthetic demos, verifies their terminal receipts,
records installed versions and hashes, and keeps logs. It does not install
dependencies, call a model API, use a GPU, acquire assets, create cloud resources,
or reproduce a real inference measurement. Its minimal subprocess environment
omits inherited credentials, but it is not an OS security sandbox.

`cpu-reproduction-record.json` is a sanitized account of one actual fresh-wheel,
outside-checkout CPU execution. Temporary local paths and full logs are omitted
from the public record. Each new reproduction should save its own receipts;
the existing record is evidence, not a command success substitute.
