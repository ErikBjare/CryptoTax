

## Create hierarchical index


```py
df = pd.DataFrame([[pd.Timestamp("2018-01-01"), 1, 103.2],
                   [pd.Timestamp("2018-01-01"), 2, 23.80],
                   [pd.Timestamp("2018-01-02"), 1, 105.1],
                   [pd.Timestamp("2018-01-02"), 2, 22.10], 
                   columns=['date', 'id', 'value']).set_index(['date', 'id'])
```

## Forward fill hierarchical data

```py
df['value'] = df.groupby("id")['value'].ffill()
```


## Get subset of DataFrame on second index

```py
df.xs(2, level="id")

df.loc[df.index.get_level_values("id") == 2]

# Can also be assigned to:
df.loc[df.index.get_level_values("id") == 2, 'value'] = 0
df.loc[df.index.get_level_values("id") == 2, 'value'] = [[103.2], [108.1]]
```

