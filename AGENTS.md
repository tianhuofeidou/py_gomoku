# Project release rules

- Every push that contains program logic changes must carry a new semantic version and matching Git tag.
- Bug fixes increment the patch number by one, for example `1.0.0` -> `1.0.1`.
- Large logic changes or newly added logic increment the minor number by one and reset the patch number to zero, for example `1.0.1` -> `1.1.0`.
- Never change the major number unless the user gives an explicit instruction to publish a new major version.
- Documentation-only or other non-logic changes do not require a version increment.
- Before pushing, keep the version shown in project documentation/metadata consistent with the Git tag wherever those files are tracked.
