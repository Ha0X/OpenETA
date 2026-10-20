---
name: openeta-configure
description: Select a registered OpenETA harness configuration for an operator-assigned benchmark task before starting a robot execution session.
---

Call `list_harness_configs` to inspect compatible configurations and their content hashes. Choose a configuration that supports the assigned embodiment and has executable capabilities with no missing requirements. Call `select_harness_config` with its exact ID and hash.

Selection binds the task, environment and seed supplied by the operator. It does not create or move a robot. A successful selection is fixed for this preparation session; execution starts in a separate session with the selected task tools. Report the selected ID after success.

If no executable compatible configuration exists, report the missing capabilities. Do not invent an ID, change the task, or claim successful setup. A stale hash requires reading the catalog again.
