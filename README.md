# TensorFlow

##Goal

Google's Tensorflow provides an introduction which dives into ML and TF together to solve a multi-feature problem — character recognition, which convolutes understanding. This attempts to overcome that by showing how to do linear regression for a single feature problem, and expand from there.


## Cheatsheet

* Linear regression: single feature, single scalar outcome
* Linear regression: multi-feature, single scalar outcome
* Logistic regression: multi-feature, multi-class outcome

## Code

* linear_regression_one_feature.py
    * ML with linear regression for a single feature
        * Example: predict house price from house size (single feature)
* linear_regression_one_feature_with_tensorboard.py
    * Add visualization for 'ML for single feature' with Tensorboard
        * Use tf.scalar_summary, tf.histogram_summary to collect data for variables that we want to visualize
        * Use `scope` to collapse TF network graph in to expandable/collapsible black boxes to faciliate visualization
* linear_regression_one_feature_using_mini_batch_with_tensorboard.py
    * Perform 'stochastic/mini-batch/batch' Gradient Descent with TF
    * The CUSTOMIZABLE section contains all the configurations that we can tweak, e.g., batch size, etc.
* linear_regression_multi_feature_using_mini_batch_without_matrix_with_tensorboard.py
    * ML with linear regrssion for 2 features without using 'matrix'
    * Create additional tf.Variable, tf.placeholder for each feature
    * **IMPORTANT**: This is a messy way to do ML with multiple features. This is provided as an explanation of multi-feature concept.
* linear_regression_multi_feature_using_mini_batch_with_tensorboard.py
    * ML with linear regrssion for 2 features
    * Expanding existing W (tf.Variable) in matrix 'height', and existing x (tf.placeholder) in matrix 'width' to accomodate each feature
