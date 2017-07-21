Heart Disease Logistic Regression Model

run cleanup.py for datawrangling.  cleanup.py writes to clean.csv with processed data. I have included my copy of clean.csv, so this step is not required
run testsep.py to define:
	doit(data, labels)  --  the function that runs the logistic regression
	yobs -- the observed y values
	hdbasic -- dataframe with basic vitals info 
	hdrestecg -- dataframe with resting ecg data
	hdlabwork -- dataframe with cholesterol and fasting blood sugar
	hdecg --  dataframe with stress ecg
	hdthal -- dataframe with thallium stress test
	hdca -- dataframe with cardiac catheterization
	hdall -- dataframe with all available features

examples:  doit(hdall, yobs) will run logistic regression on all features
	   doit(hdbasic.join(hdlabwork), yobs) will run logistic regression on basic vitals and  routine labwork