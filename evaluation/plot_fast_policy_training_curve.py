from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RAW_LOG = r"""
[Epoch] 1/100 mode=softlabel_ugw loss=1.3718 soft_ce=1.3766 imitation_loss=0.0000 entropy=0.9470 target_entropy=0.0078 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9956 align_cost=0.0887 click_rate=1.0000
[Epoch] 2/100 mode=softlabel_ugw loss=1.3615 soft_ce=1.3661 imitation_loss=0.0000 entropy=0.9274 target_entropy=0.0071 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9958 align_cost=0.0889 click_rate=1.0000
[Epoch] 3/100 mode=softlabel_ugw loss=1.3586 soft_ce=1.3632 imitation_loss=0.0000 entropy=0.9287 target_entropy=0.0075 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9958 align_cost=0.0890 click_rate=1.0000
[Epoch] 4/100 mode=softlabel_ugw loss=1.3505 soft_ce=1.3551 imitation_loss=0.0000 entropy=0.9216 target_entropy=0.0073 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9957 align_cost=0.0890 click_rate=1.0000
[Epoch] 5/100 mode=softlabel_ugw loss=1.3460 soft_ce=1.3506 imitation_loss=0.0000 entropy=0.9246 target_entropy=0.0062 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9966 align_cost=0.0890 click_rate=1.0000
[Epoch] 6/100 mode=softlabel_ugw loss=1.3469 soft_ce=1.3515 imitation_loss=0.0000 entropy=0.9156 target_entropy=0.0085 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9953 align_cost=0.0888 click_rate=1.0000
[Epoch] 7/100 mode=softlabel_ugw loss=1.3438 soft_ce=1.3484 imitation_loss=0.0000 entropy=0.9207 target_entropy=0.0079 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9955 align_cost=0.0890 click_rate=1.0000
[Epoch] 8/100 mode=softlabel_ugw loss=1.3414 soft_ce=1.3459 imitation_loss=0.0000 entropy=0.9151 target_entropy=0.0090 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9950 align_cost=0.0890 click_rate=1.0000
[Epoch] 9/100 mode=softlabel_ugw loss=1.3389 soft_ce=1.3435 imitation_loss=0.0000 entropy=0.9133 target_entropy=0.0075 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9959 align_cost=0.0890 click_rate=1.0000
[Epoch] 10/100 mode=softlabel_ugw loss=1.3359 soft_ce=1.3405 imitation_loss=0.0000 entropy=0.9056 target_entropy=0.0072 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9961 align_cost=0.0888 click_rate=1.0000
[Epoch] 11/100 mode=softlabel_ugw loss=1.3323 soft_ce=1.3369 imitation_loss=0.0000 entropy=0.9122 target_entropy=0.0074 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9958 align_cost=0.0891 click_rate=1.0000
[Epoch] 12/100 mode=softlabel_ugw loss=1.3361 soft_ce=1.3407 imitation_loss=0.0000 entropy=0.9130 target_entropy=0.0074 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9959 align_cost=0.0888 click_rate=1.0000
[Epoch] 13/100 mode=softlabel_ugw loss=1.3354 soft_ce=1.3400 imitation_loss=0.0000 entropy=0.9039 target_entropy=0.0075 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9956 align_cost=0.0890 click_rate=1.0000
[Epoch] 14/100 mode=softlabel_ugw loss=1.3423 soft_ce=1.3469 imitation_loss=0.0000 entropy=0.9141 target_entropy=0.0067 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9962 align_cost=0.0888 click_rate=1.0000
[Epoch] 15/100 mode=softlabel_ugw loss=1.3348 soft_ce=1.3394 imitation_loss=0.0000 entropy=0.9103 target_entropy=0.0081 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9955 align_cost=0.0890 click_rate=1.0000
[Epoch] 16/100 mode=softlabel_ugw loss=1.3316 soft_ce=1.3361 imitation_loss=0.0000 entropy=0.8999 target_entropy=0.0089 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9949 align_cost=0.0888 click_rate=1.0000
[Epoch] 17/100 mode=softlabel_ugw loss=1.3359 soft_ce=1.3405 imitation_loss=0.0000 entropy=0.9146 target_entropy=0.0084 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9952 align_cost=0.0890 click_rate=1.0000
[Epoch] 18/100 mode=softlabel_ugw loss=1.3323 soft_ce=1.3368 imitation_loss=0.0000 entropy=0.9073 target_entropy=0.0085 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9952 align_cost=0.0890 click_rate=1.0000
[Epoch] 19/100 mode=softlabel_ugw loss=1.3331 soft_ce=1.3377 imitation_loss=0.0000 entropy=0.9071 target_entropy=0.0091 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9949 align_cost=0.0889 click_rate=1.0000
[Epoch] 20/100 mode=softlabel_ugw loss=1.3300 soft_ce=1.3345 imitation_loss=0.0000 entropy=0.9017 target_entropy=0.0080 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9956 align_cost=0.0890 click_rate=1.0000
[Epoch] 21/100 mode=softlabel_ugw loss=1.3277 soft_ce=1.3322 imitation_loss=0.0000 entropy=0.9057 target_entropy=0.0065 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9964 align_cost=0.0890 click_rate=1.0000
[Epoch] 22/100 mode=softlabel_ugw loss=1.3352 soft_ce=1.3397 imitation_loss=0.0000 entropy=0.9086 target_entropy=0.0073 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9960 align_cost=0.0890 click_rate=1.0000
[Epoch] 23/100 mode=softlabel_ugw loss=1.3369 soft_ce=1.3415 imitation_loss=0.0000 entropy=0.9129 target_entropy=0.0081 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9954 align_cost=0.0887 click_rate=1.0000
[Epoch] 24/100 mode=softlabel_ugw loss=1.3316 soft_ce=1.3362 imitation_loss=0.0000 entropy=0.9074 target_entropy=0.0072 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9960 align_cost=0.0889 click_rate=1.0000
[Epoch] 25/100 mode=softlabel_ugw loss=1.3248 soft_ce=1.3292 imitation_loss=0.0000 entropy=0.8957 target_entropy=0.0073 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9960 align_cost=0.0889 click_rate=1.0000
[Epoch] 26/100 mode=softlabel_ugw loss=1.3258 soft_ce=1.3303 imitation_loss=0.0000 entropy=0.9002 target_entropy=0.0073 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9959 align_cost=0.0892 click_rate=1.0000
[Epoch] 27/100 mode=softlabel_ugw loss=1.3256 soft_ce=1.3301 imitation_loss=0.0000 entropy=0.9065 target_entropy=0.0073 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9960 align_cost=0.0888 click_rate=1.0000
[Epoch] 28/100 mode=softlabel_ugw loss=1.3284 soft_ce=1.3329 imitation_loss=0.0000 entropy=0.9019 target_entropy=0.0091 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9950 align_cost=0.0892 click_rate=1.0000
[Epoch] 29/100 mode=softlabel_ugw loss=1.3269 soft_ce=1.3314 imitation_loss=0.0000 entropy=0.8955 target_entropy=0.0083 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9954 align_cost=0.0891 click_rate=1.0000
[Epoch] 30/100 mode=softlabel_ugw loss=1.3315 soft_ce=1.3360 imitation_loss=0.0000 entropy=0.8997 target_entropy=0.0082 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9953 align_cost=0.0890 click_rate=1.0000
[Epoch] 31/100 mode=softlabel_ugw loss=1.3272 soft_ce=1.3317 imitation_loss=0.0000 entropy=0.9073 target_entropy=0.0078 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9956 align_cost=0.0889 click_rate=1.0000
[Epoch] 32/100 mode=softlabel_ugw loss=1.3341 soft_ce=1.3385 imitation_loss=0.0000 entropy=0.8977 target_entropy=0.0076 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9957 align_cost=0.0888 click_rate=1.0000
[Epoch] 33/100 mode=softlabel_ugw loss=1.3302 soft_ce=1.3347 imitation_loss=0.0000 entropy=0.9094 target_entropy=0.0092 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9947 align_cost=0.0890 click_rate=1.0000
[Epoch] 34/100 mode=softlabel_ugw loss=1.3335 soft_ce=1.3381 imitation_loss=0.0000 entropy=0.9028 target_entropy=0.0084 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9951 align_cost=0.0889 click_rate=1.0000
[Epoch] 35/100 mode=softlabel_ugw loss=1.3299 soft_ce=1.3344 imitation_loss=0.0000 entropy=0.9037 target_entropy=0.0091 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9946 align_cost=0.0888 click_rate=1.0000
[Epoch] 36/100 mode=softlabel_ugw loss=1.3256 soft_ce=1.3301 imitation_loss=0.0000 entropy=0.8981 target_entropy=0.0070 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9961 align_cost=0.0891 click_rate=1.0000
[Epoch] 37/100 mode=softlabel_ugw loss=1.3305 soft_ce=1.3350 imitation_loss=0.0000 entropy=0.9028 target_entropy=0.0069 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9963 align_cost=0.0890 click_rate=1.0000
[Epoch] 38/100 mode=softlabel_ugw loss=1.3331 soft_ce=1.3376 imitation_loss=0.0000 entropy=0.9094 target_entropy=0.0085 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9950 align_cost=0.0892 click_rate=1.0000
[Epoch] 39/100 mode=softlabel_ugw loss=1.3258 soft_ce=1.3303 imitation_loss=0.0000 entropy=0.8987 target_entropy=0.0081 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9955 align_cost=0.0890 click_rate=1.0000
[Epoch] 40/100 mode=softlabel_ugw loss=1.3266 soft_ce=1.3311 imitation_loss=0.0000 entropy=0.9034 target_entropy=0.0074 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9960 align_cost=0.0890 click_rate=1.0000
[Epoch] 41/100 mode=softlabel_ugw loss=1.3224 soft_ce=1.3269 imitation_loss=0.0000 entropy=0.9014 target_entropy=0.0087 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9952 align_cost=0.0889 click_rate=1.0000
[Epoch] 42/100 mode=softlabel_ugw loss=1.3240 soft_ce=1.3285 imitation_loss=0.0000 entropy=0.8998 target_entropy=0.0081 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9953 align_cost=0.0890 click_rate=1.0000
[Epoch] 43/100 mode=softlabel_ugw loss=1.3196 soft_ce=1.3241 imitation_loss=0.0000 entropy=0.8959 target_entropy=0.0070 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9960 align_cost=0.0891 click_rate=1.0000
[Epoch] 44/100 mode=softlabel_ugw loss=1.3252 soft_ce=1.3297 imitation_loss=0.0000 entropy=0.8972 target_entropy=0.0075 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9956 align_cost=0.0888 click_rate=1.0000
[Epoch] 45/100 mode=softlabel_ugw loss=1.3276 soft_ce=1.3321 imitation_loss=0.0000 entropy=0.8984 target_entropy=0.0080 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9953 align_cost=0.0888 click_rate=1.0000
[Epoch] 46/100 mode=softlabel_ugw loss=1.3243 soft_ce=1.3288 imitation_loss=0.0000 entropy=0.9049 target_entropy=0.0096 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9947 align_cost=0.0888 click_rate=1.0000
[Epoch] 47/100 mode=softlabel_ugw loss=1.3229 soft_ce=1.3274 imitation_loss=0.0000 entropy=0.8955 target_entropy=0.0078 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9957 align_cost=0.0889 click_rate=1.0000
[Epoch] 48/100 mode=softlabel_ugw loss=1.3248 soft_ce=1.3293 imitation_loss=0.0000 entropy=0.9007 target_entropy=0.0087 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9950 align_cost=0.0892 click_rate=1.0000
[Epoch] 49/100 mode=softlabel_ugw loss=1.3254 soft_ce=1.3300 imitation_loss=0.0000 entropy=0.9060 target_entropy=0.0073 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9959 align_cost=0.0889 click_rate=1.0000
[Epoch] 50/100 mode=softlabel_ugw loss=1.3246 soft_ce=1.3291 imitation_loss=0.0000 entropy=0.8988 target_entropy=0.0079 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9957 align_cost=0.0889 click_rate=1.0000
[Epoch] 51/100 mode=softlabel_ugw loss=1.3212 soft_ce=1.3257 imitation_loss=0.0000 entropy=0.9003 target_entropy=0.0090 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9950 align_cost=0.0887 click_rate=1.0000
[Epoch] 52/100 mode=softlabel_ugw loss=1.3242 soft_ce=1.3287 imitation_loss=0.0000 entropy=0.8968 target_entropy=0.0072 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9960 align_cost=0.0889 click_rate=1.0000
[Epoch] 53/100 mode=softlabel_ugw loss=1.3256 soft_ce=1.3301 imitation_loss=0.0000 entropy=0.9010 target_entropy=0.0092 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9948 align_cost=0.0887 click_rate=1.0000
[Epoch] 54/100 mode=softlabel_ugw loss=1.3279 soft_ce=1.3324 imitation_loss=0.0000 entropy=0.8958 target_entropy=0.0077 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9957 align_cost=0.0888 click_rate=1.0000
[Epoch] 55/100 mode=softlabel_ugw loss=1.3287 soft_ce=1.3332 imitation_loss=0.0000 entropy=0.9066 target_entropy=0.0070 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9960 align_cost=0.0889 click_rate=1.0000
[Epoch] 56/100 mode=softlabel_ugw loss=1.3239 soft_ce=1.3284 imitation_loss=0.0000 entropy=0.9019 target_entropy=0.0086 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9951 align_cost=0.0889 click_rate=1.0000
[Epoch] 57/100 mode=softlabel_ugw loss=1.3189 soft_ce=1.3233 imitation_loss=0.0000 entropy=0.8916 target_entropy=0.0072 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9961 align_cost=0.0889 click_rate=1.0000
[Epoch] 58/100 mode=softlabel_ugw loss=1.3177 soft_ce=1.3222 imitation_loss=0.0000 entropy=0.9012 target_entropy=0.0086 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9951 align_cost=0.0889 click_rate=1.0000
[Epoch] 59/100 mode=softlabel_ugw loss=1.3252 soft_ce=1.3297 imitation_loss=0.0000 entropy=0.8973 target_entropy=0.0097 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9943 align_cost=0.0890 click_rate=1.0000
[Epoch] 60/100 mode=softlabel_ugw loss=1.3231 soft_ce=1.3276 imitation_loss=0.0000 entropy=0.8979 target_entropy=0.0089 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9951 align_cost=0.0889 click_rate=1.0000
[Epoch] 61/100 mode=softlabel_ugw loss=1.3247 soft_ce=1.3292 imitation_loss=0.0000 entropy=0.8990 target_entropy=0.0085 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9952 align_cost=0.0889 click_rate=1.0000
[Epoch] 62/100 mode=softlabel_ugw loss=1.3177 soft_ce=1.3222 imitation_loss=0.0000 entropy=0.8913 target_entropy=0.0088 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9949 align_cost=0.0890 click_rate=1.0000
[Epoch] 63/100 mode=softlabel_ugw loss=1.3160 soft_ce=1.3205 imitation_loss=0.0000 entropy=0.9010 target_entropy=0.0078 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9958 align_cost=0.0888 click_rate=1.0000
[Epoch] 64/100 mode=softlabel_ugw loss=1.3215 soft_ce=1.3260 imitation_loss=0.0000 entropy=0.9068 target_entropy=0.0081 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9953 align_cost=0.0890 click_rate=1.0000
[Epoch] 65/100 mode=softlabel_ugw loss=1.3181 soft_ce=1.3226 imitation_loss=0.0000 entropy=0.8974 target_entropy=0.0087 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9951 align_cost=0.0888 click_rate=1.0000
[Epoch] 66/100 mode=softlabel_ugw loss=1.3217 soft_ce=1.3262 imitation_loss=0.0000 entropy=0.9010 target_entropy=0.0086 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9952 align_cost=0.0887 click_rate=1.0000
[Epoch] 67/100 mode=softlabel_ugw loss=1.3236 soft_ce=1.3280 imitation_loss=0.0000 entropy=0.8884 target_entropy=0.0078 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9955 align_cost=0.0889 click_rate=1.0000
[Epoch] 68/100 mode=softlabel_ugw loss=1.3200 soft_ce=1.3245 imitation_loss=0.0000 entropy=0.9008 target_entropy=0.0069 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9963 align_cost=0.0891 click_rate=1.0000
[Epoch] 69/100 mode=softlabel_ugw loss=1.3183 soft_ce=1.3227 imitation_loss=0.0000 entropy=0.8906 target_entropy=0.0079 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9955 align_cost=0.0889 click_rate=1.0000
[Epoch] 70/100 mode=softlabel_ugw loss=1.3242 soft_ce=1.3287 imitation_loss=0.0000 entropy=0.9004 target_entropy=0.0086 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9950 align_cost=0.0889 click_rate=1.0000
[Epoch] 71/100 mode=softlabel_ugw loss=1.3194 soft_ce=1.3238 imitation_loss=0.0000 entropy=0.8860 target_entropy=0.0077 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9958 align_cost=0.0891 click_rate=1.0000
[Epoch] 72/100 mode=softlabel_ugw loss=1.3177 soft_ce=1.3222 imitation_loss=0.0000 entropy=0.8983 target_entropy=0.0078 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9958 align_cost=0.0890 click_rate=1.0000
[Epoch] 73/100 mode=softlabel_ugw loss=1.3213 soft_ce=1.3258 imitation_loss=0.0000 entropy=0.9027 target_entropy=0.0072 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9961 align_cost=0.0889 click_rate=1.0000
[Epoch] 74/100 mode=softlabel_ugw loss=1.3186 soft_ce=1.3230 imitation_loss=0.0000 entropy=0.8960 target_entropy=0.0094 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9947 align_cost=0.0891 click_rate=1.0000
[Epoch] 75/100 mode=softlabel_ugw loss=1.3153 soft_ce=1.3197 imitation_loss=0.0000 entropy=0.8935 target_entropy=0.0089 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9949 align_cost=0.0890 click_rate=1.0000
[Epoch] 76/100 mode=softlabel_ugw loss=1.3240 soft_ce=1.3285 imitation_loss=0.0000 entropy=0.8989 target_entropy=0.0081 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9951 align_cost=0.0890 click_rate=1.0000
[Epoch] 77/100 mode=softlabel_ugw loss=1.3171 soft_ce=1.3216 imitation_loss=0.0000 entropy=0.8993 target_entropy=0.0069 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9964 align_cost=0.0889 click_rate=1.0000
[Epoch] 78/100 mode=softlabel_ugw loss=1.3242 soft_ce=1.3287 imitation_loss=0.0000 entropy=0.8959 target_entropy=0.0075 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9957 align_cost=0.0889 click_rate=1.0000
[Epoch] 79/100 mode=softlabel_ugw loss=1.3169 soft_ce=1.3214 imitation_loss=0.0000 entropy=0.8991 target_entropy=0.0084 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9952 align_cost=0.0889 click_rate=1.0000
[Epoch] 80/100 mode=softlabel_ugw loss=1.3182 soft_ce=1.3226 imitation_loss=0.0000 entropy=0.8967 target_entropy=0.0076 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9957 align_cost=0.0891 click_rate=1.0000
[Epoch] 81/100 mode=softlabel_ugw loss=1.3210 soft_ce=1.3255 imitation_loss=0.0000 entropy=0.8958 target_entropy=0.0083 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9952 align_cost=0.0889 click_rate=1.0000
[Epoch] 82/100 mode=softlabel_ugw loss=1.3155 soft_ce=1.3200 imitation_loss=0.0000 entropy=0.9018 target_entropy=0.0070 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9964 align_cost=0.0889 click_rate=1.0000
[Epoch] 83/100 mode=softlabel_ugw loss=1.3222 soft_ce=1.3267 imitation_loss=0.0000 entropy=0.8977 target_entropy=0.0065 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9964 align_cost=0.0892 click_rate=1.0000
[Epoch] 84/100 mode=softlabel_ugw loss=1.3160 soft_ce=1.3204 imitation_loss=0.0000 entropy=0.8887 target_entropy=0.0072 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9959 align_cost=0.0889 click_rate=1.0000
[Epoch] 85/100 mode=softlabel_ugw loss=1.3129 soft_ce=1.3174 imitation_loss=0.0000 entropy=0.8959 target_entropy=0.0086 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9952 align_cost=0.0890 click_rate=1.0000
[Epoch] 86/100 mode=softlabel_ugw loss=1.3139 soft_ce=1.3183 imitation_loss=0.0000 entropy=0.8836 target_entropy=0.0071 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9961 align_cost=0.0892 click_rate=1.0000
[Epoch] 87/100 mode=softlabel_ugw loss=1.3159 soft_ce=1.3204 imitation_loss=0.0000 entropy=0.8872 target_entropy=0.0078 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9957 align_cost=0.0889 click_rate=1.0000
[Epoch] 88/100 mode=softlabel_ugw loss=1.3190 soft_ce=1.3234 imitation_loss=0.0000 entropy=0.8905 target_entropy=0.0079 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9956 align_cost=0.0888 click_rate=1.0000
[Epoch] 89/100 mode=softlabel_ugw loss=1.3172 soft_ce=1.3217 imitation_loss=0.0000 entropy=0.8973 target_entropy=0.0073 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9960 align_cost=0.0888 click_rate=1.0000
[Epoch] 90/100 mode=softlabel_ugw loss=1.3166 soft_ce=1.3211 imitation_loss=0.0000 entropy=0.8953 target_entropy=0.0073 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9959 align_cost=0.0890 click_rate=1.0000
[Epoch] 91/100 mode=softlabel_ugw loss=1.3192 soft_ce=1.3237 imitation_loss=0.0000 entropy=0.9025 target_entropy=0.0080 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9959 align_cost=0.0889 click_rate=1.0000
[Epoch] 92/100 mode=softlabel_ugw loss=1.3133 soft_ce=1.3178 imitation_loss=0.0000 entropy=0.8931 target_entropy=0.0065 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9966 align_cost=0.0888 click_rate=1.0000
[Epoch] 93/100 mode=softlabel_ugw loss=1.3094 soft_ce=1.3138 imitation_loss=0.0000 entropy=0.8846 target_entropy=0.0083 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9952 align_cost=0.0889 click_rate=1.0000
[Epoch] 94/100 mode=softlabel_ugw loss=1.3197 soft_ce=1.3242 imitation_loss=0.0000 entropy=0.8977 target_entropy=0.0083 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9953 align_cost=0.0889 click_rate=1.0000
[Epoch] 95/100 mode=softlabel_ugw loss=1.3134 soft_ce=1.3179 imitation_loss=0.0000 entropy=0.9012 target_entropy=0.0075 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9958 align_cost=0.0889 click_rate=1.0000
[Epoch] 96/100 mode=softlabel_ugw loss=1.3150 soft_ce=1.3195 imitation_loss=0.0000 entropy=0.8957 target_entropy=0.0085 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9953 align_cost=0.0889 click_rate=1.0000
[Epoch] 97/100 mode=softlabel_ugw loss=1.3151 soft_ce=1.3196 imitation_loss=0.0000 entropy=0.8947 target_entropy=0.0060 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9969 align_cost=0.0887 click_rate=1.0000
[Epoch] 98/100 mode=softlabel_ugw loss=1.3146 soft_ce=1.3190 imitation_loss=0.0000 entropy=0.8855 target_entropy=0.0079 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9951 align_cost=0.0891 click_rate=1.0000
[Epoch] 99/100 mode=softlabel_ugw loss=1.3061 soft_ce=1.3106 imitation_loss=0.0000 entropy=0.8897 target_entropy=0.0085 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9953 align_cost=0.0887 click_rate=1.0000
[Epoch] 100/100 mode=softlabel_ugw loss=1.3170 soft_ce=1.3215 imitation_loss=0.0000 entropy=0.8848 target_entropy=0.0077 target_skip=0.0000 target_skip_w=0.0000 target_click_mass=0.9958 align_cost=0.0887 click_rate=1.0000
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot fast policy training loss curve.")
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/reports/fast_policy_training_curve.pdf",
        help="Path to save the generated PDF figure.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.14,
        help="Smoothing factor for the exponential moving average trend line.",
    )
    return parser.parse_args()


def parse_log(raw_log: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pattern = re.compile(
        r"\[Epoch\]\s+(\d+)/\d+.*?loss=([0-9.]+)\s+soft_ce=([0-9.]+)"
    )
    epochs: list[int] = []
    losses: list[float] = []
    soft_ce: list[float] = []
    for line in raw_log.strip().splitlines():
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            losses.append(float(match.group(2)))
            soft_ce.append(float(match.group(3)))
    if not epochs:
        raise ValueError("No epoch records were parsed from RAW_LOG.")
    return (
        np.array(epochs, dtype=np.int32),
        np.array(losses, dtype=np.float64),
        np.array(soft_ce, dtype=np.float64),
    )


def ema_curve(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    epochs, losses, soft_ce = parse_log(RAW_LOG)
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
        soft_ce,
        color="#C9D6EA",
        linewidth=1.4,
        linestyle="--",
        alpha=0.95,
        label="Soft CE",
        zorder=1,
    )
    ax.plot(
        epochs,
        losses,
        color="#9ECAE1",
        linewidth=1.8,
        alpha=0.98,
        label="Raw Loss",
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

    ax.set_title("Fast policy training curve")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_xlim(int(epochs.min()), int(epochs.max()))
    ax.margins(x=0.01)

    y_min = min(float(losses.min()), float(soft_ce.min()))
    y_max = max(float(losses.max()), float(soft_ce.max()))
    padding = (y_max - y_min) * 0.12
    ax.set_ylim(y_min - padding, y_max + padding)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")

    ax.legend(frameon=False, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(
        f"Saved figure to {output_path.resolve()} | "
        f"best_epoch={min_epoch}, min_loss={min_loss:.4f}"
    )


if __name__ == "__main__":
    main()
