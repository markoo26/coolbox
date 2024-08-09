def sparse_softmax(values, k=1):
    """
       Computes the sparse softmax of a list of values.

       Each value is raised to the power of `k` and normalized by the sum of all powered values.

       Parameters:
       -----------
       values : list of float
           List of numerical values to be normalized.
       k : float, optional
           Exponent for the values, default is 1.

       Returns:
       --------
       list of float
           Normalized values as a list.

       Example:
       --------
       >>> sparse_softmax([1.0, 2.0, 3.0], k=2)
       [0.071, 0.286, 0.643]
       """
    power_of_values = sum([value ** k for value in values])
    return [value ** k / power_of_values for value in values]
