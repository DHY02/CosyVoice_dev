import pandas as pd
df = pd.DataFrame({
    "src_audio": ["/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/exp/cosyvoice/test_init/instruct/liuchanhg_201_fear_0.wav"],
    "tgt_audio": ["/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/m3ed/test/A_guainiguofenmeili_1_15.wav"]
})
df.to_csv("input.tsv", sep="\t", index=False, encoding='utf-8')