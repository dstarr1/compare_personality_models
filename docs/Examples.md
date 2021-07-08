### Original example:
```
### encode seed text, which will be continued by our model:
###  The [BOS] and [EOS] tags mark the sentence demarcation

ids = tokenizer.encode('[BOS] The King must leave the throne now . [EOS]',
                      return_tensors='pt')
```
