install:
	python3 setup.py install
	
test_scMRA:
	python3 test/test_scMRA_OptimizationSetup.py
	diff test/data/CplexMraProblemTrue.lp test/data/CplexMraProblem.lp

test_scCNR:
	python3 test/test_scCNR_OptimizationSetup.py
	diff test/data/CplexCnrProblemTrue.lp test/data/CplexCnrProblem.lp
