# Python Environment


Create Venv
---

### For Windows

```shell=
python -m venv [env-name]
```

### For Mac

```shell=
python3 -m venv [env-name]
```

Enter Venv
---

### For Windows

```shell=
[env-name]\Scripts\activate.bat
```

### For Mac

```shell=
source [env-name]/bin/activate
```

Python Coding
---

1. enter
```shell=
python
```

2. input
```shell=
import sys
sys.path
```

3. Leave
`Ctrl + Z` & `Press Enter`

Install Package
---

### Install Package Without Version

#### For Windows
```shell=
python -m pip install novas
```

#### For Mac
```shell=
python3 -m pip install novas
```

### Install Package With Version

#### For Windows
```shell=
python -m pip install requests==2.6.0
```

#### For Mac
```shell=
python3 -m pip install requests==2.6.0
```

### Update Package

#### For Windows
```shell=
python -m pip install --upgrade requests
```

#### For Mac
```shell=
python3 -m pip install --upgrade requests
```

### Show Package Information

#### For Windows and Mac
```shell=
pip show requests
```

### Show All Package In Venv

#### For Windows and Mac
```shell=
pip list
```

### Output Package Requirement list And Show

#### For Windows and Mac

1. save
```shell=
pip freeze > requirements.txt
```

2. show
```shell=
cat requirements.txt
```

### Version Control For Requirement Loading

#### For Windows and Mac
```shell=
python -m pip install -r requirements.txt
```



###### tags: `BCD` `Python` `Model` `DeepLearning` `TeamWork`
