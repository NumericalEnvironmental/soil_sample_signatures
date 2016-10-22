# soil_sample_signatures
This is a python script I used to employ example data mining tools from scikit-learn to identify spatial variability in petroleum hydrocarbon product signatures in soil samples collected from a large environmental site. It serves as a simple demo as to how to use both pandas and scikit-learn to easily sort through a medium-sized data set and glean insights.

The following tab-delimited input files are required:

* fuel_ref.txt = reference compositions for petroleum products (by TPH range); several examples are included for each product so that averages and standard deviations can be computed
* survey.txt = soil boring survey information (location ID, northing, easting, and surface elevation); example file not currently provided (to protect anonymity of the site)
* soil_samples.txt = soil chemistry data (location ID that matching survey data file, depth, date, name of analyte, result); example file not currently provided (to protect anonymity of the site)

More background information can be found here: https://numericalenvironmental.wordpress.com/2016/08/17/classifiers-for-distributed-soil-samples/

Email me with questions at walt.mcnab@gmail.com. 
