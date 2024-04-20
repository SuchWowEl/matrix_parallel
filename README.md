# Installation
Make sure to install beforehand `pipenv`

```
$ python3 -m venv .venv
$ source .venv/bin/activate or ./venv/Scripts/activate.ps1
$ pipenv install
```

# To execute
Activate the `venv` as stated in step 2, then

`$ mpiexec -n 4 python __main__.py`
