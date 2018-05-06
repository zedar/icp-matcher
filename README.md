The matching of features uses ICP (Iterative Closest Point) algorithm.

A feature is point alike object with geometry (3D) and descriptive attributes (like category, heading/orientation).

The result of matching X to Y contains:

1. status if X has been matched to Y with the given error
2. transformed X's geometry (first 3 features) as well as transformation matrix (including rotation and translation)

The shape of X comparing to Y should meet the following conditions:

* number of X's rows <= Y's number of rows
* number of X's columns == Y's number of columns