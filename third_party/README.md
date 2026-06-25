# third_party/

Vendored open-source code (e.g. `dreamerv3` and friends) plus the thin adapters
that bridge it to our `crafter_rl.env.CrafterGym` and logging conventions. Keep
each upstream project in its own subdirectory with its original license intact;
keep our glue code minimal and clearly separated from the vendored sources.

Anything reusable that we write on top should graduate into the `crafter_rl`
package rather than living here. The promotion / vendoring workflow is
documented separately (docs to come).
