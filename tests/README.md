# Computational Checks

From the repository root, run:

```bash
python -m unittest discover -s tests -v
```

Fourteen checks exercise synthetic inputs and circuit identities. Eleven checks
require the original empirical inputs or saved results and explicitly skip
when those files are absent. The full local author package runs all 25.

The quickstart notebook is also executed separately to validate its reading
and computation sequence. These checks assess implementation consistency.
