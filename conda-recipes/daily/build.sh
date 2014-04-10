base=`dirname $RECIPE_DIR`
base=`dirname $base`
cp -r $base .
cd `basename $base`
$PYTHON setup.py version
$PYTHON setup.py install --single-version-externally-managed --record=record.txt