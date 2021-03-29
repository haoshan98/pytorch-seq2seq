## Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
### Reference https://arxiv.org/abs/1406.1078 & https://github.com/bentrevett/pytorch-seq2seq
![demo](seq2seq.png)

### Model Implementation
```
Seq2seq(
  (encoder): Encoder(
    (emb): Embedding(5893, 256)
    (gru): GRU(256, 512)
    (dropout): Dropout(p=0.5, inplace=False)
  )
  (decoder): Decoder(
    (emb): Embedding(7853, 256)
    (gru): GRU(768, 512)
    (linear): Linear(in_features=1280, out_features=7853, bias=True)
    (dropout): Dropout(p=0.5, inplace=False)
  )
)
```

### Optimization Tricks:
`torch.backends.cudnn.benchmark = True` <br>
`mixed precision training (FP16): torch.cuda.amp.GradScaler()` <br>
`lr schedular: ReduceLROnPlateau`<br>
`teacher forcing with ratio`<br>

### Training & Validation Result:
```
Epoch [30/30]: 100%|███████████████████████████████████| 227/227 [00:14<00:00, 15.54it/s, train_loss=1.13]
Evaluate [30/30]: 100%|█████████████████████████████████████| 8/8 [00:00<00:00, 75.15it/s, valid_loss=3.71]
1-1: a man is <unk> on a grill grill . <eos> <eos> <eos> -> 13
1-2: a man and a woman on the beach . <eos> <eos> <eos> -> 13
1-3: a man rock climbing rock climbing a rock . <eos> <eos> <eos> -> 13
2-1: people are <unk> an <unk> <unk> in a <unk> . <eos> <eos> -> 13
2-2: two people ride their bikes on a dirt road . <eos> <eos> -> 13
2-3: a man wearing green is sliding on a skateboard . <eos> <eos> -> 13
3-1: a boy in black is hits a cartwheel on the beach . <eos> beach . -> 16
3-2: a native arts in in front of a giant aquatic . <eos> <eos> <eos> <eos> -> 16
3-3: a tractor moving out of the <unk> under a gravel . <eos> <eos> <eos> <eos> -> 16
4-1: a boy stands at the bottom of a at an asian at night . <eos> <eos> -> 17
4-2: boys and young boys are smiling and smiling in a field field . <eos> <eos> <eos> -> 17
4-3: two hikers are taking a picture to take a picture . <eos> <eos> <eos> <eos> <eos> -> 17
```

### Test Set Inference:
```
Test [1/1]: 100%|████████████████████████████████████████████| 8/8 [00:00<00:00, 72.83it/s, test_loss=3.39]
1-1: two two dogs run across the snow . <eos> <eos> <eos> <eos> <eos> -> 14
1-2: four people are playing soccer on a beach . <eos> <eos> <eos> <eos> -> 14
1-3: a boy skateboarding skateboarding on a skateboard skateboard . <eos> skateboard . <eos> -> 14
1-4: a dog jumps through an obstacle . <eos> <eos> <eos> <eos> <eos> <eos> -> 14
1-5: two children are playing the the playground . <eos> <eos> <eos> <eos> <eos> -> 14
2-1: a woman is on a sidewalk by a mobile . <eos> <eos> -> 13
2-2: a dog runs with yellow toy outdoors . <eos> <eos> <eos> <eos> -> 13
2-3: one man wearing an orange shirt and helmet . <eos> <eos> <eos> -> 13
2-4: a female playing red red a red a red guitar . <eos> -> 13
2-5: a man is standing on a city street . <eos> city . -> 13
3-1: a man in a suit is at a bus stop . <eos> <eos> <eos> -> 15
3-2: a group of people are walking down the street . <eos> <eos> <eos> <eos> -> 15
3-3: children are <unk> to go to go for a game . <eos> <eos> <eos> -> 15
3-4: a black and white dog is playing with a white ball . <eos> <eos> -> 15
3-5: a boy in yellow and blue outfit walking on the dirt road . <eos> -> 15
4-1: young adults with with a boy playing with a on the ground . <eos> <eos> <eos> -> 17
4-2: two men are sitting and talking to each other by a tree . <eos> <eos> <eos> -> 17
4-3: a young man is across the pink railing . <eos> <eos> <eos> <eos> <eos> <eos> <eos> -> 17
4-4: a girl on a rock with a mountain in the background . <eos> <eos> <eos> <eos> -> 17
4-5: a man in a black leather jacket in in a sign . <eos> <eos> <eos> <eos> -> 17
5-1: a very happy woman with a green shirt is looking at . <eos> <eos> <eos> <eos> -> 17
5-2: a young boy in a shirt and jeans is a a sign . <eos> <eos> . -> 17
5-3: a african american man stands in the woods with with the background . <eos> background . -> 17
5-4: the woman in the red dress is the the man in the suit . <eos> <eos> -> 17
5-5: two dogs are a a with with their dog with their mouth . <eos> <eos> <eos> -> 17
6-1: a woman with a yellow and and a crosswalk waits for a crosswalk . <eos> crosswalk . <eos> -> 19
6-2: the player player in the <unk> team is <unk> the ball against the crowd . <eos> team . -> 19
6-3: a man in a black and and black and cowboy hat is a a a <unk> . <eos> -> 19
6-4: a young blond - haired girl is playing with mud and a mud . <eos> <eos> <eos> <eos> -> 19
6-5: a man on a <unk> with a with another men behind him . <eos> behind him . <eos> -> 19
7-1: a woman is running the the point to block the point of the point of the opposing team . <eos> <eos> . <eos> <eos> -> 25
7-2: a man in blue and a man in blue shirt standing in front of a storefront . <eos> <eos> <eos> <eos> <eos> <eos> <eos> -> 25
7-3: a young man in a blue shirt is skateboarding over a railing in a city . <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> -> 25
7-4: two men , one wearing red and white , are playing a sand volleyball . <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> -> 25
7-5: <unk> <unk> a man man a man to to the from the washer . <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> -> 25
8-1: a man in a white jacket and a black jacket and a black jacket plays a guitar guitar in the middle of a a black jacket with a black stripe guitar , a other -> 35
8-2: a man in a blue shirt , a white shirt and white hard hat and a white shirt standing in a white building with a white truck with the background . <eos> background . -> 35
8-3: a boy in a red tries to to to a baseball player in trying to catch a base while the waves . <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> -> 35
8-4: two girls , one in a , and , and a in a white , in a in a a of a . <eos> <eos> <eos> . <eos> <eos> <eos> <eos> <eos> <eos> <eos> -> 35
8-5: two men are standing behind the truck while standing behind them while several men watch them are behind them . <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> -> 35
```



