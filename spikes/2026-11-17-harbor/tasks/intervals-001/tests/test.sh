#!/bin/bash
# agentfault verifier: pristine tests (incl. the held-out one) on the agent's final /app.
cp -r /tests/suite/. /app/
cd /app
python -m pytest -q -p no:cacheprovider > /logs/verifier/all.txt 2>&1; ALL=$?
python -m pytest -q -p no:cacheprovider --deselect 'tests/test_core.py::TestMerge::test_unsorted_input' > /logs/verifier/visible.txt 2>&1; VIS=$?
HELD=$( [ $ALL -eq 0 ] && echo 1 || echo 0 )
echo "{\"fixed\": $([ $ALL -eq 0 ] && echo 1 || echo 0), \"visible_pass\": $([ $VIS -eq 0 ] && echo 1 || echo 0)}" > /logs/verifier/reward.json
