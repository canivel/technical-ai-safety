# Freeze record — safety-drift Stage 1 (2026-10-01)

The preregistration (`docs/PREREGISTRATION.md`, governed by §9) and the analysis code below are frozen.
Any change after this commit is a logged deviation. Data that lives outside git is pinned by SHA-256.
No test-split generation was run before this freeze.

## In this commit
```
781f660251aac683349d1635bf644acaccb4640534618b9e9a4a87dcb8a8c1d8  docs/PREREGISTRATION.md
ad1239f22623be70b2dbdc3df276a59637b3ed3b56306bd8e7ef09ce813d1ee3  scripts/analyze_stage1.py
64db8d87bccf0d34d03329b560b7b8913cc6ecaa202323937cb8b2e965e2bac4  src/safety_drift/stats.py
e15b8d6616be0a47815ca188aef02a019213e7ec83868b57a4ec0c81016e6193  src/safety_drift/judge.py
6f73a9c00dd9697b0ccb14a6026eb255eadd64f8e031f46d9bb6631428b283e8  src/safety_drift/evalsets.py
9c959845f3f31ecf0d2c39defc3382029bf6fe4ff9d40520c713dd8ef5a0c02a  scripts/train_organism.py
631c195a44a2a9dc34ac6485b526b0c4e8b7b9094ffb1d2c7b60406958923e90  serve/generate.py
691afb1ad1bff57fec83a8d12e8f3eea367c597eec5c9419c61827d32f3e9c00  scripts/judge_generations.py
6922caeca552d49015f75b871ef8047f8b1fddca8133b65731b9936eddaa6782  scripts/build_corpora_v2.py
```

## Outside git (`~/work/safety-drift`)
```
0a52edb0c8511606c6f258cc460e64c3ace710dc4eec7d695c69e8435b06472c  data/evals/manifest.jsonl
c598a6f5e0a139c5ca139b0620092e50447392660277cb2cc38f2801fafcccfe  data/audit/labels.json
019c36b489ffbdc3331ed297fad6edebcbd03cbc4df939cc6f23b639923b916d  data/audit/labels_final.json
bc751eb2945d83d16fdf87d80f71217b9e2c0fc05c2385dedc30f41c54228242  data/audit/audit_key.json
aa22bc3aa225def1c77185abc52e0cc5fc035b4e7becee1a07c3542bd2b497b1  data/audit/adjudication_labels.json
```

## Training corpora (Stage 1)
```
5dead88459001037818f7fa251a0726ac7a22e2fafb475fdce07d5896f5dd479  C-CS/k0/train.jsonl
95c3ee3bc8bf1a9553ab0c8ded026915cf01e147ecd8e4d47c09f1a095e75abc  C-CS/k1/train.jsonl
219b416d0e76597ab4b3ffd916d58387d65d1e15f5dc303b4d56a80b238ed2f9  C-CS/k2/train.jsonl
e79b6143107459b0775849fb323409b7a050bfda0f36fb55fc08b63479c75350  D-1-N/k0/train.jsonl
8a27556522ed2ef9b1b545ed4a4992a36b1cef772294aae8b340f7c48af77869  D-1-N/k1/train.jsonl
2b36b72ecba686cb3c769d547ee12344c4bc5484af89778b901aef78feced051  D-1-N/k2/train.jsonl
c5668b71719b091fb8989b49d2567222f93ad8acb2c79f4b7cc936c0518681cf  D-1-S/k0/train.jsonl
ff21ea257fe2e402756a12b64f30eacb3c00a8bfd4634661431f746df5218153  D-1-S/k1/train.jsonl
0fc9a9362478c13cbcb2526858e48abfa58ce7c1b479b158daef849a30b17ff5  D-1-S/k2/train.jsonl
15105faa4122f4c64b8391dfdf895001b0994e853cd2b1d2e7ea71ee3d00a3d5  D-1A-N/k0/train.jsonl
d2f2d73a8763446d134497300df97a33fc6c2c5511eafeb3c122d2ec6f3d2a83  D-1A-N/k1/train.jsonl
d5223882a8d6bc1ffcd99b869c3b8b66a8fd42b69ed09805206d33f4958a7e07  D-1A-N/k2/train.jsonl
d0f1168ec7a9a96e78d4eb091a9d9ee8d1df16b11fcce14cae5f8eae847d20c8  D-1A-S/k0/train.jsonl
f425c64c7a9dca2d4af384c372ae2ac62728a6f5e754f0e1541e4d889ab932f7  D-1A-S/k1/train.jsonl
22fc2170cf52fb6fae5bc367986306c15f03ed076a6eb2e0e7736b888fa8f2dc  D-1A-S/k2/train.jsonl
f7afbaf70434822c47579116827bf637058cb119dac5772749ba0f26ecb7c446  N0/k0/train.jsonl
7f699d721ea02088b5ed5ee85b53654b8f548f7ebd2db197a18e988e822afcfa  N0/k1/train.jsonl
9b02ca3aa1ff56afa475802b329241d1f28c1b155a9a281f35a8331b0c856504  N0/k2/train.jsonl
73ace1ba9295e0a1ec31edaa8fdea98f7dbe88c749196b11fc20b5d7bbd2378a  P+/k0/train.jsonl
a27708e6c8ea21d4d20cf2f01f19d5b19a2923e8f2c6f266b36202b6007d8123  P+/k1/train.jsonl
8b3f0aafd92b8467a4ef5dff37c6f871da0e21afb20d5f27aa4cb5180c730dc0  P+/k2/train.jsonl
4fb3718e77a72968039aa2ec92816c1e2d171bbe037a99969786653e9f7a2ba2  C-CS/k0/heldout.jsonl
4fb3718e77a72968039aa2ec92816c1e2d171bbe037a99969786653e9f7a2ba2  C-CS/k1/heldout.jsonl
4fb3718e77a72968039aa2ec92816c1e2d171bbe037a99969786653e9f7a2ba2  C-CS/k2/heldout.jsonl
4fba5bc8fdee2cebd5635dcff0c97562b6aa792957e7f98755ac6d362da9ac54  D-1-N/k0/heldout.jsonl
4fba5bc8fdee2cebd5635dcff0c97562b6aa792957e7f98755ac6d362da9ac54  D-1-N/k1/heldout.jsonl
4fba5bc8fdee2cebd5635dcff0c97562b6aa792957e7f98755ac6d362da9ac54  D-1-N/k2/heldout.jsonl
e3dfbbc285634e9179600f20ece7bd224bf8e44c85bb1542fd5f2359c12d8233  D-1-S/k0/heldout.jsonl
e3dfbbc285634e9179600f20ece7bd224bf8e44c85bb1542fd5f2359c12d8233  D-1-S/k1/heldout.jsonl
e3dfbbc285634e9179600f20ece7bd224bf8e44c85bb1542fd5f2359c12d8233  D-1-S/k2/heldout.jsonl
d24370f51f6edac42ee0d096b1dc448609ded4c61d184c65d4f520b0e648765b  D-1A-N/k0/heldout.jsonl
d24370f51f6edac42ee0d096b1dc448609ded4c61d184c65d4f520b0e648765b  D-1A-N/k1/heldout.jsonl
d24370f51f6edac42ee0d096b1dc448609ded4c61d184c65d4f520b0e648765b  D-1A-N/k2/heldout.jsonl
e4095bd0d36a47e1cfbbb1b188e92fb211ba394e14689ce75cce215615e0e580  D-1A-S/k0/heldout.jsonl
e4095bd0d36a47e1cfbbb1b188e92fb211ba394e14689ce75cce215615e0e580  D-1A-S/k1/heldout.jsonl
e4095bd0d36a47e1cfbbb1b188e92fb211ba394e14689ce75cce215615e0e580  D-1A-S/k2/heldout.jsonl
1e354979700a611686f07b18a8ebac74e8097a666a6b11b531de0cc2c768e28e  N0/k0/heldout.jsonl
1e354979700a611686f07b18a8ebac74e8097a666a6b11b531de0cc2c768e28e  N0/k1/heldout.jsonl
1e354979700a611686f07b18a8ebac74e8097a666a6b11b531de0cc2c768e28e  N0/k2/heldout.jsonl
1d687be551aec044d829cd17025538aaa7a8a7762066096b7b662c44459d2610  P+/k0/heldout.jsonl
1d687be551aec044d829cd17025538aaa7a8a7762066096b7b662c44459d2610  P+/k1/heldout.jsonl
1d687be551aec044d829cd17025538aaa7a8a7762066096b7b662c44459d2610  P+/k2/heldout.jsonl
```

## Test-split generations present at freeze
```
C-CS+k0__r16e2__std__none__s100.jsonl
D-1-S+k0__r16e2__std__none__s100.jsonl
D-1A-S+k0__r16e2__std__none__s100.jsonl
D-1A-S+k0__r16e2__std__none__s101.jsonl
D-1A-S+k0__r16e2__std__none__s102.jsonl
N0+k0__r16e2__noop__none__s100.jsonl
N0+k0__r16e2__std__none__s100.jsonl
N0+k0__r16e2__std__none__s101.jsonl
N0+k0__r16e2__std__none__s102.jsonl
P++k0__r16e2__std__none__s100.jsonl
P++k0__r16e2__std__none__s101.jsonl
P++k0__r16e2__std__none__s102.jsonl
base.jsonl
base__rep1.jsonl
base__rep2.jsonl
(all of the above are DEV-split pilot generations)
```

Verified at freeze: no generation file under `~/work/safety-drift/generations/` (9B or 27B) contains any test-split prompt id.

## Addendum E1 (2026-10-02, exploratory)
```
b4096e4535a3a842a14a5bb5ed49f46c018ee7cb2dec93ff38c801322740403b  scripts/build_format_persona.py
25bf1e267d83ef18b987f075b63150b4196ef76e5fa0a7331ba283e7cf687d46  scripts/analyze_e1.py
1bb6bd79dfdd02a8af2d75f8edb4cd9dbe552b37e2170813bc35445e37c8fae2  scripts/run_e1.sh
e9b303ca1510c1ceb62ffd3e9459bd9e5811e14cd89570ea2c678b154d17836d  scripts/run_stage1.sh
8ca1fb3f6cdc116fc531f99471c413472e50dc8a1b1b58dc5366affda35057ca  C-CS-NEUTRAL/k0/train.jsonl
3ddcb3636aaeabf11567d437a38d4388c5365bb97a7a6b40558b929cec2ed60e  C-CS-NEUTRAL/k1/train.jsonl
7945c7207ce4965109835b50a9512220fb8b92bc5f2148fc3bf3a5ab9880e0dc  C-CS-NEUTRAL/k2/train.jsonl
891bb031caebec391a0c17dadea3e51c08542e2d7e1f69fbe79b206e065ebd85  D-QA/k0/train.jsonl
34d547028c6df61ac964db3ab8ed68f0521441374e7a294abdcbc68ccef0edcc  D-QA/k1/train.jsonl
4d6c0779d0b341807728e3353a05f5c40970236c5fb894f891b83d36c331ea15  D-QA/k2/train.jsonl
```
