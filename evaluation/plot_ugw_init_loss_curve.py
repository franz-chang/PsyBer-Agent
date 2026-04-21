from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RAW_LOG = r"""
[UGW-Init] epoch=1/500 mse=0.163076
[UGW-Init] epoch=2/500 mse=0.115712
[UGW-Init] epoch=3/500 mse=0.043149
[UGW-Init] epoch=4/500 mse=0.012313
[UGW-Init] epoch=5/500 mse=0.017334
[UGW-Init] epoch=6/500 mse=0.019918
[UGW-Init] epoch=7/500 mse=0.020219
[UGW-Init] epoch=8/500 mse=0.018319
[UGW-Init] epoch=9/500 mse=0.013066
[UGW-Init] epoch=10/500 mse=0.014648
[UGW-Init] epoch=11/500 mse=0.011578
[UGW-Init] epoch=12/500 mse=0.012510
[UGW-Init] epoch=13/500 mse=0.012844
[UGW-Init] epoch=14/500 mse=0.010755
[UGW-Init] epoch=15/500 mse=0.009639
[UGW-Init] epoch=16/500 mse=0.009089
[UGW-Init] epoch=17/500 mse=0.008537
[UGW-Init] epoch=18/500 mse=0.008873
[UGW-Init] epoch=19/500 mse=0.008920
[UGW-Init] epoch=20/500 mse=0.008113
[UGW-Init] epoch=21/500 mse=0.008064
[UGW-Init] epoch=22/500 mse=0.008218
[UGW-Init] epoch=23/500 mse=0.007712
[UGW-Init] epoch=24/500 mse=0.007803
[UGW-Init] epoch=25/500 mse=0.007884
[UGW-Init] epoch=26/500 mse=0.007572
[UGW-Init] epoch=27/500 mse=0.007430
[UGW-Init] epoch=28/500 mse=0.007434
[UGW-Init] epoch=29/500 mse=0.007277
[UGW-Init] epoch=30/500 mse=0.007148
[UGW-Init] epoch=31/500 mse=0.007185
[UGW-Init] epoch=32/500 mse=0.007004
[UGW-Init] epoch=33/500 mse=0.006911
[UGW-Init] epoch=34/500 mse=0.006776
[UGW-Init] epoch=35/500 mse=0.006730
[UGW-Init] epoch=36/500 mse=0.006596
[UGW-Init] epoch=37/500 mse=0.006443
[UGW-Init] epoch=38/500 mse=0.006297
[UGW-Init] epoch=39/500 mse=0.006323
[UGW-Init] epoch=40/500 mse=0.006051
[UGW-Init] epoch=41/500 mse=0.005886
[UGW-Init] epoch=42/500 mse=0.005713
[UGW-Init] epoch=43/500 mse=0.005504
[UGW-Init] epoch=44/500 mse=0.005411
[UGW-Init] epoch=45/500 mse=0.005292
[UGW-Init] epoch=46/500 mse=0.005168
[UGW-Init] epoch=47/500 mse=0.005059
[UGW-Init] epoch=48/500 mse=0.005035
[UGW-Init] epoch=49/500 mse=0.004908
[UGW-Init] epoch=50/500 mse=0.004812
[UGW-Init] epoch=51/500 mse=0.004810
[UGW-Init] epoch=52/500 mse=0.004721
[UGW-Init] epoch=53/500 mse=0.004692
[UGW-Init] epoch=54/500 mse=0.004555
[UGW-Init] epoch=55/500 mse=0.004476
[UGW-Init] epoch=56/500 mse=0.004431
[UGW-Init] epoch=57/500 mse=0.004430
[UGW-Init] epoch=58/500 mse=0.004347
[UGW-Init] epoch=59/500 mse=0.004250
[UGW-Init] epoch=60/500 mse=0.004190
[UGW-Init] epoch=61/500 mse=0.004096
[UGW-Init] epoch=62/500 mse=0.003929
[UGW-Init] epoch=63/500 mse=0.003835
[UGW-Init] epoch=64/500 mse=0.003637
[UGW-Init] epoch=65/500 mse=0.003563
[UGW-Init] epoch=66/500 mse=0.003507
[UGW-Init] epoch=67/500 mse=0.003459
[UGW-Init] epoch=68/500 mse=0.003403
[UGW-Init] epoch=69/500 mse=0.003309
[UGW-Init] epoch=70/500 mse=0.003240
[UGW-Init] epoch=71/500 mse=0.003172
[UGW-Init] epoch=72/500 mse=0.003174
[UGW-Init] epoch=73/500 mse=0.003135
[UGW-Init] epoch=74/500 mse=0.003105
[UGW-Init] epoch=75/500 mse=0.003091
[UGW-Init] epoch=76/500 mse=0.003115
[UGW-Init] epoch=77/500 mse=0.003051
[UGW-Init] epoch=78/500 mse=0.003028
[UGW-Init] epoch=79/500 mse=0.002978
[UGW-Init] epoch=80/500 mse=0.002984
[UGW-Init] epoch=81/500 mse=0.003000
[UGW-Init] epoch=82/500 mse=0.002965
[UGW-Init] epoch=83/500 mse=0.002943
[UGW-Init] epoch=84/500 mse=0.002918
[UGW-Init] epoch=85/500 mse=0.002900
[UGW-Init] epoch=86/500 mse=0.002910
[UGW-Init] epoch=87/500 mse=0.002866
[UGW-Init] epoch=88/500 mse=0.002857
[UGW-Init] epoch=89/500 mse=0.002838
[UGW-Init] epoch=90/500 mse=0.002822
[UGW-Init] epoch=91/500 mse=0.002806
[UGW-Init] epoch=92/500 mse=0.002787
[UGW-Init] epoch=93/500 mse=0.002739
[UGW-Init] epoch=94/500 mse=0.002743
[UGW-Init] epoch=95/500 mse=0.002715
[UGW-Init] epoch=96/500 mse=0.002714
[UGW-Init] epoch=97/500 mse=0.002657
[UGW-Init] epoch=98/500 mse=0.002676
[UGW-Init] epoch=99/500 mse=0.002648
[UGW-Init] epoch=100/500 mse=0.002662
[UGW-Init] epoch=101/500 mse=0.002642
[UGW-Init] epoch=102/500 mse=0.002609
[UGW-Init] epoch=103/500 mse=0.002628
[UGW-Init] epoch=104/500 mse=0.002621
[UGW-Init] epoch=105/500 mse=0.002606
[UGW-Init] epoch=106/500 mse=0.002572
[UGW-Init] epoch=107/500 mse=0.002544
[UGW-Init] epoch=108/500 mse=0.002558
[UGW-Init] epoch=109/500 mse=0.002525
[UGW-Init] epoch=110/500 mse=0.002491
[UGW-Init] epoch=111/500 mse=0.002500
[UGW-Init] epoch=112/500 mse=0.002486
[UGW-Init] epoch=113/500 mse=0.002477
[UGW-Init] epoch=114/500 mse=0.002463
[UGW-Init] epoch=115/500 mse=0.002439
[UGW-Init] epoch=116/500 mse=0.002429
[UGW-Init] epoch=117/500 mse=0.002422
[UGW-Init] epoch=118/500 mse=0.002416
[UGW-Init] epoch=119/500 mse=0.002415
[UGW-Init] epoch=120/500 mse=0.002416
[UGW-Init] epoch=121/500 mse=0.002398
[UGW-Init] epoch=122/500 mse=0.002397
[UGW-Init] epoch=123/500 mse=0.002382
[UGW-Init] epoch=124/500 mse=0.002364
[UGW-Init] epoch=125/500 mse=0.002335
[UGW-Init] epoch=126/500 mse=0.002350
[UGW-Init] epoch=127/500 mse=0.002327
[UGW-Init] epoch=128/500 mse=0.002296
[UGW-Init] epoch=129/500 mse=0.002278
[UGW-Init] epoch=130/500 mse=0.002280
[UGW-Init] epoch=131/500 mse=0.002248
[UGW-Init] epoch=132/500 mse=0.002260
[UGW-Init] epoch=133/500 mse=0.002220
[UGW-Init] epoch=134/500 mse=0.002228
[UGW-Init] epoch=135/500 mse=0.002228
[UGW-Init] epoch=136/500 mse=0.002206
[UGW-Init] epoch=137/500 mse=0.002179
[UGW-Init] epoch=138/500 mse=0.002176
[UGW-Init] epoch=139/500 mse=0.002185
[UGW-Init] epoch=140/500 mse=0.002159
[UGW-Init] epoch=141/500 mse=0.002182
[UGW-Init] epoch=142/500 mse=0.002180
[UGW-Init] epoch=143/500 mse=0.002148
[UGW-Init] epoch=144/500 mse=0.002126
[UGW-Init] epoch=145/500 mse=0.002155
[UGW-Init] epoch=146/500 mse=0.002118
[UGW-Init] epoch=147/500 mse=0.002128
[UGW-Init] epoch=148/500 mse=0.002155
[UGW-Init] epoch=149/500 mse=0.002105
[UGW-Init] epoch=150/500 mse=0.002108
[UGW-Init] epoch=151/500 mse=0.002083
[UGW-Init] epoch=152/500 mse=0.002086
[UGW-Init] epoch=153/500 mse=0.002088
[UGW-Init] epoch=154/500 mse=0.002044
[UGW-Init] epoch=155/500 mse=0.002062
[UGW-Init] epoch=156/500 mse=0.002033
[UGW-Init] epoch=157/500 mse=0.002040
[UGW-Init] epoch=158/500 mse=0.002035
[UGW-Init] epoch=159/500 mse=0.002010
[UGW-Init] epoch=160/500 mse=0.002056
[UGW-Init] epoch=161/500 mse=0.002017
[UGW-Init] epoch=162/500 mse=0.002012
[UGW-Init] epoch=163/500 mse=0.001991
[UGW-Init] epoch=164/500 mse=0.002034
[UGW-Init] epoch=165/500 mse=0.001984
[UGW-Init] epoch=166/500 mse=0.001971
[UGW-Init] epoch=167/500 mse=0.001988
[UGW-Init] epoch=168/500 mse=0.001955
[UGW-Init] epoch=169/500 mse=0.001955
[UGW-Init] epoch=170/500 mse=0.002000
[UGW-Init] epoch=171/500 mse=0.001947
[UGW-Init] epoch=172/500 mse=0.001965
[UGW-Init] epoch=173/500 mse=0.001985
[UGW-Init] epoch=174/500 mse=0.001960
[UGW-Init] epoch=175/500 mse=0.002009
[UGW-Init] epoch=176/500 mse=0.001981
[UGW-Init] epoch=177/500 mse=0.001955
[UGW-Init] epoch=178/500 mse=0.001956
[UGW-Init] epoch=179/500 mse=0.001928
[UGW-Init] epoch=180/500 mse=0.001940
[UGW-Init] epoch=181/500 mse=0.001909
[UGW-Init] epoch=182/500 mse=0.001921
[UGW-Init] epoch=183/500 mse=0.001942
[UGW-Init] epoch=184/500 mse=0.001888
[UGW-Init] epoch=185/500 mse=0.001910
[UGW-Init] epoch=186/500 mse=0.001887
[UGW-Init] epoch=187/500 mse=0.001891
[UGW-Init] epoch=188/500 mse=0.001892
[UGW-Init] epoch=189/500 mse=0.001879
[UGW-Init] epoch=190/500 mse=0.001872
[UGW-Init] epoch=191/500 mse=0.001845
[UGW-Init] epoch=192/500 mse=0.001869
[UGW-Init] epoch=193/500 mse=0.001853
[UGW-Init] epoch=194/500 mse=0.001856
[UGW-Init] epoch=195/500 mse=0.001851
[UGW-Init] epoch=196/500 mse=0.001849
[UGW-Init] epoch=197/500 mse=0.001830
[UGW-Init] epoch=198/500 mse=0.001838
[UGW-Init] epoch=199/500 mse=0.001840
[UGW-Init] epoch=200/500 mse=0.001823
[UGW-Init] epoch=201/500 mse=0.001842
[UGW-Init] epoch=202/500 mse=0.001825
[UGW-Init] epoch=203/500 mse=0.001829
[UGW-Init] epoch=204/500 mse=0.001836
[UGW-Init] epoch=205/500 mse=0.001833
[UGW-Init] epoch=206/500 mse=0.001805
[UGW-Init] epoch=207/500 mse=0.001812
[UGW-Init] epoch=208/500 mse=0.001808
[UGW-Init] epoch=209/500 mse=0.001824
[UGW-Init] epoch=210/500 mse=0.001798
[UGW-Init] epoch=211/500 mse=0.001798
[UGW-Init] epoch=212/500 mse=0.001797
[UGW-Init] epoch=213/500 mse=0.001802
[UGW-Init] epoch=214/500 mse=0.001799
[UGW-Init] epoch=215/500 mse=0.001772
[UGW-Init] epoch=216/500 mse=0.001778
[UGW-Init] epoch=217/500 mse=0.001811
[UGW-Init] epoch=218/500 mse=0.001809
[UGW-Init] epoch=219/500 mse=0.001777
[UGW-Init] epoch=220/500 mse=0.001782
[UGW-Init] epoch=221/500 mse=0.001799
[UGW-Init] epoch=222/500 mse=0.001779
[UGW-Init] epoch=223/500 mse=0.001769
[UGW-Init] epoch=224/500 mse=0.001747
[UGW-Init] epoch=225/500 mse=0.001759
[UGW-Init] epoch=226/500 mse=0.001740
[UGW-Init] epoch=227/500 mse=0.001763
[UGW-Init] epoch=228/500 mse=0.001731
[UGW-Init] epoch=229/500 mse=0.001749
[UGW-Init] epoch=230/500 mse=0.001730
[UGW-Init] epoch=231/500 mse=0.001754
[UGW-Init] epoch=232/500 mse=0.001768
[UGW-Init] epoch=233/500 mse=0.001754
[UGW-Init] epoch=234/500 mse=0.001770
[UGW-Init] epoch=235/500 mse=0.001792
[UGW-Init] epoch=236/500 mse=0.001742
[UGW-Init] epoch=237/500 mse=0.001759
[UGW-Init] epoch=238/500 mse=0.001763
[UGW-Init] epoch=239/500 mse=0.001785
[UGW-Init] epoch=240/500 mse=0.001726
[UGW-Init] epoch=241/500 mse=0.001737
[UGW-Init] epoch=242/500 mse=0.001726
[UGW-Init] epoch=243/500 mse=0.001738
[UGW-Init] epoch=244/500 mse=0.001710
[UGW-Init] epoch=245/500 mse=0.001713
[UGW-Init] epoch=246/500 mse=0.001774
[UGW-Init] epoch=247/500 mse=0.001753
[UGW-Init] epoch=248/500 mse=0.001723
[UGW-Init] epoch=249/500 mse=0.001733
[UGW-Init] epoch=250/500 mse=0.001726
[UGW-Init] epoch=251/500 mse=0.001717
[UGW-Init] epoch=252/500 mse=0.001689
[UGW-Init] epoch=253/500 mse=0.001692
[UGW-Init] epoch=254/500 mse=0.001711
[UGW-Init] epoch=255/500 mse=0.001684
[UGW-Init] epoch=256/500 mse=0.001696
[UGW-Init] epoch=257/500 mse=0.001697
[UGW-Init] epoch=258/500 mse=0.001680
[UGW-Init] epoch=259/500 mse=0.001670
[UGW-Init] epoch=260/500 mse=0.001671
[UGW-Init] epoch=261/500 mse=0.001662
[UGW-Init] epoch=262/500 mse=0.001655
[UGW-Init] epoch=263/500 mse=0.001661
[UGW-Init] epoch=264/500 mse=0.001658
[UGW-Init] epoch=265/500 mse=0.001649
[UGW-Init] epoch=266/500 mse=0.001633
[UGW-Init] epoch=267/500 mse=0.001645
[UGW-Init] epoch=268/500 mse=0.001652
[UGW-Init] epoch=269/500 mse=0.001636
[UGW-Init] epoch=270/500 mse=0.001638
[UGW-Init] epoch=271/500 mse=0.001662
[UGW-Init] epoch=272/500 mse=0.001630
[UGW-Init] epoch=273/500 mse=0.001623
[UGW-Init] epoch=274/500 mse=0.001638
[UGW-Init] epoch=275/500 mse=0.001627
[UGW-Init] epoch=276/500 mse=0.001618
[UGW-Init] epoch=277/500 mse=0.001615
[UGW-Init] epoch=278/500 mse=0.001635
[UGW-Init] epoch=279/500 mse=0.001619
[UGW-Init] epoch=280/500 mse=0.001620
[UGW-Init] epoch=281/500 mse=0.001609
[UGW-Init] epoch=282/500 mse=0.001608
[UGW-Init] epoch=283/500 mse=0.001612
[UGW-Init] epoch=284/500 mse=0.001592
[UGW-Init] epoch=285/500 mse=0.001605
[UGW-Init] epoch=286/500 mse=0.001593
[UGW-Init] epoch=287/500 mse=0.001609
[UGW-Init] epoch=288/500 mse=0.001584
[UGW-Init] epoch=289/500 mse=0.001593
[UGW-Init] epoch=290/500 mse=0.001577
[UGW-Init] epoch=291/500 mse=0.001589
[UGW-Init] epoch=292/500 mse=0.001590
[UGW-Init] epoch=293/500 mse=0.001582
[UGW-Init] epoch=294/500 mse=0.001586
[UGW-Init] epoch=295/500 mse=0.001579
[UGW-Init] epoch=296/500 mse=0.001580
[UGW-Init] epoch=297/500 mse=0.001568
[UGW-Init] epoch=298/500 mse=0.001577
[UGW-Init] epoch=299/500 mse=0.001582
[UGW-Init] epoch=300/500 mse=0.001590
[UGW-Init] epoch=301/500 mse=0.001561
[UGW-Init] epoch=302/500 mse=0.001579
[UGW-Init] epoch=303/500 mse=0.001562
[UGW-Init] epoch=304/500 mse=0.001560
[UGW-Init] epoch=305/500 mse=0.001568
[UGW-Init] epoch=306/500 mse=0.001552
[UGW-Init] epoch=307/500 mse=0.001547
[UGW-Init] epoch=308/500 mse=0.001548
[UGW-Init] epoch=309/500 mse=0.001550
[UGW-Init] epoch=310/500 mse=0.001546
[UGW-Init] epoch=311/500 mse=0.001548
[UGW-Init] epoch=312/500 mse=0.001531
[UGW-Init] epoch=313/500 mse=0.001535
[UGW-Init] epoch=314/500 mse=0.001546
[UGW-Init] epoch=315/500 mse=0.001534
[UGW-Init] epoch=316/500 mse=0.001534
[UGW-Init] epoch=317/500 mse=0.001532
[UGW-Init] epoch=318/500 mse=0.001536
[UGW-Init] epoch=319/500 mse=0.001538
[UGW-Init] epoch=320/500 mse=0.001547
[UGW-Init] epoch=321/500 mse=0.001528
[UGW-Init] epoch=322/500 mse=0.001530
[UGW-Init] epoch=323/500 mse=0.001530
[UGW-Init] epoch=324/500 mse=0.001613
[UGW-Init] epoch=325/500 mse=0.001539
[UGW-Init] epoch=326/500 mse=0.001540
[UGW-Init] epoch=327/500 mse=0.001525
[UGW-Init] epoch=328/500 mse=0.001554
[UGW-Init] epoch=329/500 mse=0.001511
[UGW-Init] epoch=330/500 mse=0.001536
[UGW-Init] epoch=331/500 mse=0.001502
[UGW-Init] epoch=332/500 mse=0.001526
[UGW-Init] epoch=333/500 mse=0.001509
[UGW-Init] epoch=334/500 mse=0.001497
[UGW-Init] epoch=335/500 mse=0.001494
[UGW-Init] epoch=336/500 mse=0.001505
[UGW-Init] epoch=337/500 mse=0.001492
[UGW-Init] epoch=338/500 mse=0.001497
[UGW-Init] epoch=339/500 mse=0.001496
[UGW-Init] epoch=340/500 mse=0.001481
[UGW-Init] epoch=341/500 mse=0.001482
[UGW-Init] epoch=342/500 mse=0.001500
[UGW-Init] epoch=343/500 mse=0.001493
[UGW-Init] epoch=344/500 mse=0.001570
[UGW-Init] epoch=345/500 mse=0.001503
[UGW-Init] epoch=346/500 mse=0.001620
[UGW-Init] epoch=347/500 mse=0.001507
[UGW-Init] epoch=348/500 mse=0.001487
[UGW-Init] epoch=349/500 mse=0.001497
[UGW-Init] epoch=350/500 mse=0.001479
[UGW-Init] epoch=351/500 mse=0.001524
[UGW-Init] epoch=352/500 mse=0.001479
[UGW-Init] epoch=353/500 mse=0.001536
[UGW-Init] epoch=354/500 mse=0.001476
[UGW-Init] epoch=355/500 mse=0.001544
[UGW-Init] epoch=356/500 mse=0.001463
[UGW-Init] epoch=357/500 mse=0.001490
[UGW-Init] epoch=358/500 mse=0.001458
[UGW-Init] epoch=359/500 mse=0.001448
[UGW-Init] epoch=360/500 mse=0.001442
[UGW-Init] epoch=361/500 mse=0.001440
[UGW-Init] epoch=362/500 mse=0.001443
[UGW-Init] epoch=363/500 mse=0.001438
[UGW-Init] epoch=364/500 mse=0.001445
[UGW-Init] epoch=365/500 mse=0.001437
[UGW-Init] epoch=366/500 mse=0.001442
[UGW-Init] epoch=367/500 mse=0.001449
[UGW-Init] epoch=368/500 mse=0.001435
[UGW-Init] epoch=369/500 mse=0.001447
[UGW-Init] epoch=370/500 mse=0.001469
[UGW-Init] epoch=371/500 mse=0.001440
[UGW-Init] epoch=372/500 mse=0.001499
[UGW-Init] epoch=373/500 mse=0.001416
[UGW-Init] epoch=374/500 mse=0.001420
[UGW-Init] epoch=375/500 mse=0.001413
[UGW-Init] epoch=376/500 mse=0.001410
[UGW-Init] epoch=377/500 mse=0.001404
[UGW-Init] epoch=378/500 mse=0.001411
[UGW-Init] epoch=379/500 mse=0.001398
[UGW-Init] epoch=380/500 mse=0.001395
[UGW-Init] epoch=381/500 mse=0.001409
[UGW-Init] epoch=382/500 mse=0.001449
[UGW-Init] epoch=383/500 mse=0.001408
[UGW-Init] epoch=384/500 mse=0.001464
[UGW-Init] epoch=385/500 mse=0.001409
[UGW-Init] epoch=386/500 mse=0.001454
[UGW-Init] epoch=387/500 mse=0.001396
[UGW-Init] epoch=388/500 mse=0.001404
[UGW-Init] epoch=389/500 mse=0.001404
[UGW-Init] epoch=390/500 mse=0.001432
[UGW-Init] epoch=391/500 mse=0.001385
[UGW-Init] epoch=392/500 mse=0.001401
[UGW-Init] epoch=393/500 mse=0.001457
[UGW-Init] epoch=394/500 mse=0.001419
[UGW-Init] epoch=395/500 mse=0.001409
[UGW-Init] epoch=396/500 mse=0.001406
[UGW-Init] epoch=397/500 mse=0.001432
[UGW-Init] epoch=398/500 mse=0.001406
[UGW-Init] epoch=399/500 mse=0.001396
[UGW-Init] epoch=400/500 mse=0.001478
[UGW-Init] epoch=401/500 mse=0.001389
[UGW-Init] epoch=402/500 mse=0.001386
[UGW-Init] epoch=403/500 mse=0.001381
[UGW-Init] epoch=404/500 mse=0.001359
[UGW-Init] epoch=405/500 mse=0.001359
[UGW-Init] epoch=406/500 mse=0.001360
[UGW-Init] epoch=407/500 mse=0.001349
[UGW-Init] epoch=408/500 mse=0.001360
[UGW-Init] epoch=409/500 mse=0.001351
[UGW-Init] epoch=410/500 mse=0.001353
[UGW-Init] epoch=411/500 mse=0.001348
[UGW-Init] epoch=412/500 mse=0.001350
[UGW-Init] epoch=413/500 mse=0.001363
[UGW-Init] epoch=414/500 mse=0.001407
[UGW-Init] epoch=415/500 mse=0.001349
[UGW-Init] epoch=416/500 mse=0.001371
[UGW-Init] epoch=417/500 mse=0.001345
[UGW-Init] epoch=418/500 mse=0.001359
[UGW-Init] epoch=419/500 mse=0.001362
[UGW-Init] epoch=420/500 mse=0.001392
[UGW-Init] epoch=421/500 mse=0.001370
[UGW-Init] epoch=422/500 mse=0.001335
[UGW-Init] epoch=423/500 mse=0.001363
[UGW-Init] epoch=424/500 mse=0.001320
[UGW-Init] epoch=425/500 mse=0.001334
[UGW-Init] epoch=426/500 mse=0.001318
[UGW-Init] epoch=427/500 mse=0.001326
[UGW-Init] epoch=428/500 mse=0.001320
[UGW-Init] epoch=429/500 mse=0.001313
[UGW-Init] epoch=430/500 mse=0.001319
[UGW-Init] epoch=431/500 mse=0.001308
[UGW-Init] epoch=432/500 mse=0.001308
[UGW-Init] epoch=433/500 mse=0.001295
[UGW-Init] epoch=434/500 mse=0.001296
[UGW-Init] epoch=435/500 mse=0.001307
[UGW-Init] epoch=436/500 mse=0.001300
[UGW-Init] epoch=437/500 mse=0.001311
[UGW-Init] epoch=438/500 mse=0.001322
[UGW-Init] epoch=439/500 mse=0.001306
[UGW-Init] epoch=440/500 mse=0.001374
[UGW-Init] epoch=441/500 mse=0.001314
[UGW-Init] epoch=442/500 mse=0.001344
[UGW-Init] epoch=443/500 mse=0.001331
[UGW-Init] epoch=444/500 mse=0.001435
[UGW-Init] epoch=445/500 mse=0.001407
[UGW-Init] epoch=446/500 mse=0.001334
[UGW-Init] epoch=447/500 mse=0.001390
[UGW-Init] epoch=448/500 mse=0.001327
[UGW-Init] epoch=449/500 mse=0.001308
[UGW-Init] epoch=450/500 mse=0.001292
[UGW-Init] epoch=451/500 mse=0.001281
[UGW-Init] epoch=452/500 mse=0.001284
[UGW-Init] epoch=453/500 mse=0.001268
[UGW-Init] epoch=454/500 mse=0.001278
[UGW-Init] epoch=455/500 mse=0.001266
[UGW-Init] epoch=456/500 mse=0.001286
[UGW-Init] epoch=457/500 mse=0.001275
[UGW-Init] epoch=458/500 mse=0.001274
[UGW-Init] epoch=459/500 mse=0.001264
[UGW-Init] epoch=460/500 mse=0.001263
[UGW-Init] epoch=461/500 mse=0.001259
[UGW-Init] epoch=462/500 mse=0.001254
[UGW-Init] epoch=463/500 mse=0.001245
[UGW-Init] epoch=464/500 mse=0.001252
[UGW-Init] epoch=465/500 mse=0.001253
[UGW-Init] epoch=466/500 mse=0.001242
[UGW-Init] epoch=467/500 mse=0.001244
[UGW-Init] epoch=468/500 mse=0.001245
[UGW-Init] epoch=469/500 mse=0.001242
[UGW-Init] epoch=470/500 mse=0.001241
[UGW-Init] epoch=471/500 mse=0.001243
[UGW-Init] epoch=472/500 mse=0.001240
[UGW-Init] epoch=473/500 mse=0.001243
[UGW-Init] epoch=474/500 mse=0.001232
[UGW-Init] epoch=475/500 mse=0.001244
[UGW-Init] epoch=476/500 mse=0.001236
[UGW-Init] epoch=477/500 mse=0.001222
[UGW-Init] epoch=478/500 mse=0.001246
[UGW-Init] epoch=479/500 mse=0.001232
[UGW-Init] epoch=480/500 mse=0.001235
[UGW-Init] epoch=481/500 mse=0.001238
[UGW-Init] epoch=482/500 mse=0.001219
[UGW-Init] epoch=483/500 mse=0.001218
[UGW-Init] epoch=484/500 mse=0.001206
[UGW-Init] epoch=485/500 mse=0.001209
[UGW-Init] epoch=486/500 mse=0.001261
[UGW-Init] epoch=487/500 mse=0.001223
[UGW-Init] epoch=488/500 mse=0.001230
[UGW-Init] epoch=489/500 mse=0.001209
[UGW-Init] epoch=490/500 mse=0.001225
[UGW-Init] epoch=491/500 mse=0.001229
[UGW-Init] epoch=492/500 mse=0.001193
[UGW-Init] epoch=493/500 mse=0.001204
[UGW-Init] epoch=494/500 mse=0.001196
[UGW-Init] epoch=495/500 mse=0.001198
[UGW-Init] epoch=496/500 mse=0.001191
[UGW-Init] epoch=497/500 mse=0.001213
[UGW-Init] epoch=498/500 mse=0.001261
[UGW-Init] epoch=499/500 mse=0.001201
[UGW-Init] epoch=500/500 mse=0.001272
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the UGW initializer MSE curve.")
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/reports/ugw_init_loss_curve.pdf",
        help="Path to save the generated PDF figure.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.12,
        help="Smoothing factor for the exponential moving average trend line.",
    )
    return parser.parse_args()


def parse_log(raw_log: str) -> tuple[np.ndarray, np.ndarray]:
    pattern = re.compile(r"epoch=(\d+)/\d+\s+mse=([0-9.]+)")
    epochs: list[int] = []
    losses: list[float] = []
    for line in raw_log.strip().splitlines():
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            losses.append(float(match.group(2)))
    if not epochs:
        raise ValueError("No epoch-loss pairs were parsed from RAW_LOG.")
    return np.array(epochs, dtype=np.int32), np.array(losses, dtype=np.float64)


def ema_curve(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    epochs, losses = parse_log(RAW_LOG)
    smooth_losses = ema_curve(losses, alpha=float(args.ema_alpha))

    min_idx = int(np.argmin(losses))
    min_epoch = int(epochs[min_idx])
    min_loss = float(losses[min_idx])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.2), facecolor="white")
    ax.set_facecolor("#FCFCFC")
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.24)

    ax.plot(
        epochs,
        losses,
        color="#9ECAE1",
        linewidth=1.6,
        alpha=0.95,
        label="Raw MSE",
        zorder=2,
    )
    ax.plot(
        epochs,
        smooth_losses,
        color="#1F4E79",
        linewidth=2.6,
        label="EMA Trend",
        zorder=3,
    )
    ax.scatter(
        [min_epoch],
        [min_loss],
        s=52,
        color="#D62728",
        edgecolors="white",
        linewidths=0.6,
        zorder=4,
        label=f"Best Epoch = {min_epoch}",
    )

    ax.set_title("Offline regression prior model training")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE Loss")
    ax.set_xlim(1, int(epochs.max()))
    ax.margins(x=0.01)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")

    ax.legend(frameon=False, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path.resolve()}")


if __name__ == "__main__":
    main()
