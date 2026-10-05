# Data Access and Publication Record

## Confirmed Author Statements

On 1 October 2026 the author identified Bloomberg as the upstream provider and
expressed an intention to provide Python-readable daily/weekly data on GitHub.
The author subsequently supplied https://github.com/gonghui945 as a temporary
entry point and reported confirmation from co-author Francesca Medda.

On 5 October the author requested a separate repository for this companion.
The selected address is https://github.com/gonghui945/quantum-finance-practice.
RELEASE_STATUS.json records whether publication has been verified.
No exact code-licence name or text has been supplied. Data permissions are separate.

## Selected Public Route

The public package contains code, tests, deterministic synthetic CSVs, a
quickstart notebook, schemas and reconstruction instructions. The five
market-workflow notebooks have cleared outputs. Algorithmic checks and the
quickstart run without market access. Bloomberg-derived tables, predictions,
fitted arrays and executed market outputs are held separately.

Empirical reproduction is conditional on authorised access to the same
processed extract. RECONSTRUCTION.md specifies the implemented transformation
and execution contract, including the limits of independently reconstructing
the original vendor export. A subscription or a matching file schema alone
does not establish exact input identity.

The complete author-review package is retained locally. It is not the public
upload candidate. No raw intraday archive is included in either package.

## Remaining Release Metadata

1. Supply the exact approved code-licence text.
2. Record a fixed release tag and commit; add an archival DOI only after a
   real deposit exists. A DOI is not a prerequisite for local reproduction.
3. Establish the authorised route for readers needing the exact empirical inputs,
   including the remaining provider field/export and adjustment metadata.
4. If a later data release is desired, confirm permission for that specific
   extract and its derived outputs separately.
5. Verify hashes and execution records for the chosen release, and ensure
   notebooks do not disclose restricted data in saved outputs or attachments.

The publication flag is updated only after a verified upload. A complete
exact-vendor acquisition recipe still requires the missing export metadata.
