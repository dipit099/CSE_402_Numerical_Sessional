# Base Numerical Method Codes

This folder contains four simple exam-ready Python codes:

1. `bisection_method.py`
2. `false_position_method.py`
3. `newton_raphson_method.py`
4. `bairstow_method.py`

Each file has an example problem at the top. For your exam, usually you only
need to change:

- the function `f(x)`
- the derivative `df(x)` for Newton-Raphson
- the initial interval or initial guess
- the polynomial coefficients for Bairstow

Run any file like this:

```bash
python3 bisection_method.py
```

## Error Formula Used

For iterative methods:

```text
ea = abs((new_value - old_value) / new_value) * 100
```

The code stops when:

```text
ea <= es
```

where `es` is the required error tolerance in percent.

