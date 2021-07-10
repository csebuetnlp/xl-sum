# XL-Sum

This repository contains the code, data, and models of the paper titled [**"XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages"**](http://arxiv.org/abs/2106.13822) published in *Findings of the Association for Computational Linguistics: ACL 2021.*

## Table of Contents

- [XL-Sum](#xl-sum)
  - [Table of Contents](#table-of-contents)
  - [Datasets](#datasets)
  - [Models](#models)
  - [Benchmarks](#benchmarks)
  - [Multilingual ROUGE](#multilingual-rouge)
  - [Training & Evaluation](#training--evaluation)
  - [License](#license)
  - [Citation](#citation)


## Datasets
  
  ***Disclaimer: You must agree to the [license](#license) and terms of use before using the dataset.***
  
  We are releasing two versions of the dataset: an older version that has been reported in the paper; and a newer version with **another added language (Traditional Chinese)**, more data, better formatting, better extraction, larger evaluation splits, and deduplication. We recommend using the latter and thus have organized the repository with data counts and benchmarks of the newer version. The new version contains a total of **1.35 million** article-summary pairs, making XL-Sum the **largest** text summarization dataset publicly available.
  
  All dataset files are in `.jsonl` format i.e. one JSON per line. One example from the english dataset is given below in JSON format. The fields are self-explanatory.  
  ```
  {
    "id": "technology-17657859",
    "url": "https://www.bbc.com/news/technology-17657859",
    "title": "Yahoo files e-book advert system patent applications",
    "summary": "Yahoo has signalled it is investigating e-book adverts as a way to stimulate its earnings.",
    "text": "Yahoo's patents suggest users could weigh the type of ads against the sizes of discount before purchase. It says in two US patent applications that ads for digital book readers have been \"less than optimal\" to date. The filings suggest that users could be offered titles at a variety of prices depending on the ads' prominence They add that the products shown could be determined by the type of book being read, or even the contents of a specific chapter, phrase or word. The paperwork was published by the US Patent and Trademark Office late last week and relates to work carried out at the firm's headquarters in Sunnyvale, California. \"Greater levels of advertising, which may be more valuable to an advertiser and potentially more distracting to an e-book reader, may warrant higher discounts,\" it states. Free books It suggests users could be offered ads as hyperlinks based within the book's text, in-laid text or even \"dynamic content\" such as video. Another idea suggests boxes at the bottom of a page could trail later chapters or quotes saying \"brought to you by Company A\". It adds that the more willing the customer is to see the ads, the greater the potential discount. \"Higher frequencies... may even be great enough to allow the e-book to be obtained for free,\" it states. The authors write that the type of ad could influence the value of the discount, with \"lower class advertising... such as teeth whitener advertisements\" offering a cheaper price than \"high\" or \"middle class\" adverts, for things like pizza. The inventors also suggest that ads could be linked to the mood or emotional state the reader is in as a they progress through a title. For example, they say if characters fall in love or show affection during a chapter, then ads for flowers or entertainment could be triggered. The patents also suggest this could applied to children's books - giving the Tom Hanks animated film Polar Express as an example. It says a scene showing a waiter giving the protagonists hot drinks \"may be an excellent opportunity to show an advertisement for hot cocoa, or a branded chocolate bar\". Another example states: \"If the setting includes young characters, a Coke advertisement could be provided, inviting the reader to enjoy a glass of Coke with his book, and providing a graphic of a cool glass.\" It adds that such targeting could be further enhanced by taking account of previous titles the owner has bought. 'Advertising-free zone' At present, several Amazon and Kobo e-book readers offer full-screen adverts when the device is switched off and show smaller ads on their menu screens, but the main text of the titles remains free of marketing. Yahoo does not currently provide ads to these devices, and a move into the area could boost its shrinking revenues. However, Philip Jones, deputy editor of the Bookseller magazine, said that the internet firm might struggle to get some of its ideas adopted. \"This has been mooted before and was fairly well decried,\" he said. \"Perhaps in a limited context it could work if the merchandise was strongly related to the title and was kept away from the text. \"But readers - particularly parents - like the fact that reading is an advertising-free zone. Authors would also want something to say about ads interrupting their narrative flow.\""
}
  ```

[Download](https://docs.google.com/uc?export=download&id=1fKxf9jAj0KptzlxUsI3jDbp4XLv_piiD) the complete dataset. See the [legacy](legacy/) section for the older version(s).
 
We used a 80%-10%-10% split for all languages with a few exceptions. `English` was split 93%-3.5%-3.5% for the evaluation set size to resemble that of `CNN/DM` and `XSum`; `Scottish Gaelic`, `Kyrgyz` and `Sinhala` had relatively fewer samples, their evaluation sets were increased to 500 samples for more reliable evaluation. Same articles were used for evaluation in the two variants of Chinese and Serbian to prevent data leakage in multilingual training. Individual dataset download links with train-dev-test example counts are given below:

Language      | ISO 639-1 Code | BBC subdomain(s) | Train | Dev | Test | Total | Link
--------------|----------------|------------------|-------|-----|------|-------|-----
Amharic | am | https://www.bbc.com/amharic | 5761 | 719 | 719 | 7199 | [Download](https://docs.google.com/uc?export=download&id=1RVRaSdwjuILTYFez-Nl73UMv2aubHzD6)
Arabic | ar | https://www.bbc.com/arabic | 37519 | 4689 | 4689 | 46897 | [Download](https://docs.google.com/uc?export=download&id=1lot6kJ6TPCHuBI6Ky_RQPdYpWaQCk-jl)
Azerbaijani | az | https://www.bbc.com/azeri | 6478 | 809 | 809 | 8096 | [Download](https://docs.google.com/uc?export=download&id=1DXMWzrWPwA3_bA3MA8s173tcYl-7viq9)
Bengali | bn | https://www.bbc.com/bengali | 8102 | 1012 | 1012 | 10126 | [Download](https://docs.google.com/uc?export=download&id=1h3GY8Pk1xV3DWo3Ewc9ZJQ4bU7tCS_1R)
Burmese | my | https://www.bbc.com/burmese | 4569 | 570 | 570 | 5709 | [Download](https://docs.google.com/uc?export=download&id=1PqmC8MAUVi9KSenxmlcfFuH0i3VjEk9L)
Chinese (Simplified) | zh-CN | https://www.bbc.com/ukchina/simp, https://www.bbc.com/zhongwen/simp | 37362 | 4670 | 4670 | 46702 | [Download](https://docs.google.com/uc?export=download&id=18lXt8-QTuowGfaRH8UcAilt-Uehpyhsv)
Chinese (Traditional) | zh-TW | https://www.bbc.com/ukchina/trad, https://www.bbc.com/zhongwen/trad | 37373 | 4670 | 4670 | 46713 | [Download](https://docs.google.com/uc?export=download&id=1j6ln7dwmUwWOiN2SCfIoLA5B-L8ZaiJj)
English | en | https://www.bbc.com/english, https://www.bbc.com/sinhala `*` | 306522 | 11535 | 11535 | 329592 | [Download](https://docs.google.com/uc?export=download&id=1KlTW4WTHzDdmigZnqBLdRTCkamISortQ)
French | fr | https://www.bbc.com/afrique | 8697 | 1086 | 1086 | 10869 | [Download](https://docs.google.com/uc?export=download&id=1YtC1tzCrwHrAcPCU1VOefChl7-IEYTWF)
Gujarati | gu | https://www.bbc.com/gujarati | 9119 | 1139 | 1139 | 11397 | [Download](https://docs.google.com/uc?export=download&id=1IJdTIR_Im2Saa_F2tW5UNnU2g_dWp1wG)
Hausa | ha | https://www.bbc.com/hausa | 6418 | 802 | 802 | 8022 | [Download](https://docs.google.com/uc?export=download&id=1lMkb_gYpwzd32_-waG_eNaWeehm0OpGZ)
Hindi | hi | https://www.bbc.com/hindi | 70778 | 8847 | 8847 | 88472 | [Download](https://docs.google.com/uc?export=download&id=1H3PxMwEFyzNxGXpM0KMPOkt4UcdHbiky)
Igbo | ig | https://www.bbc.com/igbo | 4183 | 522 | 522 | 5227 | [Download](https://docs.google.com/uc?export=download&id=1B5td0FABADD3xAEIWO_-ZwBuoV5kW85h)
Indonesian | id | https://www.bbc.com/indonesia | 38242 | 4780 | 4780 | 47802 | [Download](https://docs.google.com/uc?export=download&id=1FV5o-ZV3mGqGpBQYx_IAEXknmdWpMyNR)
Japanese | ja | https://www.bbc.com/japanese | 7113 | 889 | 889 | 8891 | [Download](https://docs.google.com/uc?export=download&id=1Y5Wk4wI1lrhmy-qRoAF8Ygm0wIu9mHkG)
Kirundi | rn | https://www.bbc.com/gahuza | 5746 | 718 | 718 | 7182 | [Download](https://docs.google.com/uc?export=download&id=1DbPJGYoGAfclcvgWTgf0YP48u6gBdS-t)
Korean | ko | https://www.bbc.com/korean | 4407 | 550 | 550 | 5507 | [Download](https://docs.google.com/uc?export=download&id=1F_f8LonURmklzHH24B_DgFHWsiFps7x0)
Kyrgyz | ky | https://www.bbc.com/kyrgyz | 2266 | 500 | 500 | 3266 | [Download](https://docs.google.com/uc?export=download&id=19YTnLF_m8gCElwMcNhY1VGhznbq8ur5d)
Marathi | mr | https://www.bbc.com/marathi | 10903 | 1362 | 1362 | 13627 | [Download](https://docs.google.com/uc?export=download&id=1WJNQ5PqqM4FPq7VSWezx1-OFOlZ9dUiU)
Nepali | np | https://www.bbc.com/nepali | 5808 | 725 | 725 | 7258 | [Download](https://docs.google.com/uc?export=download&id=1o_T_Chy-p-Sn2AnSlvaf97nEyV1A3Tfz)
Oromo | om | https://www.bbc.com/afaanoromoo | 6063 | 757 | 757 | 7577 | [Download](https://docs.google.com/uc?export=download&id=1ZCfG5L8A77P4BvOVm3O7BXQFliGyQCqY)
Pashto | ps | https://www.bbc.com/pashto | 14353 | 1794 | 1794 | 17941 | [Download](https://docs.google.com/uc?export=download&id=1_zQzLVQgEb7fg4A-NU17C0OBi-W0DQLT)
Persian | fa | https://www.bbc.com/persian | 47251 | 5906 | 5906 | 59063 | [Download](https://docs.google.com/uc?export=download&id=1bymTi8KKSB3qZKB1qejEGyOE8LCdEvhn)
Pidgin`**` | n/a | https://www.bbc.com/pidgin | 9208 | 1151 | 1151 | 11510 | [Download](https://docs.google.com/uc?export=download&id=17n8UpuZSbWisvkPB7eklM3VARN3iNO6y)
Portuguese | pt | https://www.bbc.com/portuguese | 57402 | 7175 | 7175 | 71752 | [Download](https://docs.google.com/uc?export=download&id=1mwUJtxgDjm-ZerXWusdMc19qYtsgepBT)
Punjabi | pa | https://www.bbc.com/punjabi | 8215 | 1026 | 1026 | 10267 | [Download](https://docs.google.com/uc?export=download&id=1w6yuGHeZOQ-nhZQRd_a7oIpItNY553tL)
Russian | ru | https://www.bbc.com/russian, https://www.bbc.com/ukrainian `*` | 62243 | 7780 | 7780 | 77803 | [Download](https://docs.google.com/uc?export=download&id=1bGzbX_zYvuCHeT__7LBlG6hkAQ-dfThY)
Scottish Gaelic | gd | https://www.bbc.com/naidheachdan | 1313 | 500 | 500 | 2313 | [Download](https://docs.google.com/uc?export=download&id=1lOHjY8IcnGbrPjia5dD2sT84DYLIytT_)
Serbian (Cyrillic) | sr | https://www.bbc.com/serbian/cyr | 7275 | 909 | 909 | 9093 | [Download](https://docs.google.com/uc?export=download&id=1c_vaD4ydnTcn0pqYMbbVjt7EPIgwfUv9)
Serbian (Latin) | sr | https://www.bbc.com/serbian/lat | 7276 | 909 | 909 | 9094 | [Download](https://docs.google.com/uc?export=download&id=1JlDU401_3XqbmpaaJnvv7S1aIBd-GoqG)
Sinhala | si | https://www.bbc.com/sinhala | 3249 | 500 | 500 | 4249 | [Download](https://docs.google.com/uc?export=download&id=1HBSvf7T5qh8ox6C7lO1DqKTBVda0vUAo)
Somali | so | https://www.bbc.com/somali | 5962 | 745 | 745 | 7452 | [Download](https://docs.google.com/uc?export=download&id=1f9_4DjgTxmfhquJqnSe2x170FILsAiYi)
Spanish | es | https://www.bbc.com/mundo | 38110 | 4763 | 4763 | 47636 | [Download](https://docs.google.com/uc?export=download&id=1DDhUTm0cijq4Tx9isaG5AFT9viexBroO)
Swahili | sw | https://www.bbc.com/swahili | 7898 | 987 | 987 | 9872 | [Download](https://docs.google.com/uc?export=download&id=1WitOEsFdJdirZYZr92H0v0HGFpwi2vbh)
Tamil | ta | https://www.bbc.com/tamil | 16222 | 2027 | 2027 | 20276 | [Download](https://docs.google.com/uc?export=download&id=1ukjkPZktUBvckWliCSotUZYXBalZ3t7h)
Telugu | te | https://www.bbc.com/telugu | 10421 | 1302 | 1302 | 13025 | [Download](https://docs.google.com/uc?export=download&id=1cTbqTwYPu5U09U1mBVIN3b71W4B17gOl)
Thai | th | https://www.bbc.com/thai | 6616 | 826 | 826 | 8268 | [Download](https://docs.google.com/uc?export=download&id=1bg2pFl2YSWH90J1ll7ZMFHOnaXJ0tejF)
Tigrinya | ti | https://www.bbc.com/tigrinya | 5451 | 681 | 681 | 6813 | [Download](https://docs.google.com/uc?export=download&id=13Ob4gkiswGPEjj4iTphZD_irpFAcK0P-)
Turkish | tr | https://www.bbc.com/turkce | 27176 | 3397 | 3397 | 33970 | [Download](https://docs.google.com/uc?export=download&id=1AcnfIw-MnoNq-kpON74RW5DuJBu9MBh8)
Ukrainian | uk | https://www.bbc.com/ukrainian | 43201 | 5399 | 5399 | 53999 | [Download](https://docs.google.com/uc?export=download&id=1t5cnjvu3rEhz_LDTPvjTD5EQp5fveeOQ)
Urdu | ur | https://www.bbc.com/urdu | 67665 | 8458 | 8458 | 84581 | [Download](https://docs.google.com/uc?export=download&id=1Vie5jfHyHBkkW6jLbFNU5qjStcHstOKn)
Uzbek | uz | https://www.bbc.com/uzbek | 4728 | 590 | 590 | 5908 | [Download](https://docs.google.com/uc?export=download&id=1FK-TSViqsfBKX8bAGCUD1IlrSirlVwuS)
Vietnamese | vi | https://www.bbc.com/vietnamese | 32111 | 4013 | 4013 | 40137 | [Download](https://docs.google.com/uc?export=download&id=1ufC9hPtC-gYNTI9TCXrVudhIXGOFs-al)
Welsh | cy | https://www.bbc.com/cymrufyw | 9732 | 1216 | 1216 | 12164 | [Download](https://docs.google.com/uc?export=download&id=1mMF9I-KxO83ktJ98aVSRC30bnMTthVV6)
Yoruba | yo | https://www.bbc.com/yoruba | 6350 | 793 | 793 | 7936 | [Download](https://docs.google.com/uc?export=download&id=1oHxPjk7PQ0JSkbf906wZYyK4pGdEulLL)

`*` A lot of articles in BBC Sinhala and BBC Ukrainian were written in English and Russian respectively. They were identified using [Fasttext](https://arxiv.org/abs/1607.01759) and moved accordingly.

`**` West African Pidgin English


## Models
  We are releasing a [multilingual model checkpoint](https://docs.google.com/uc?export=download&id=137wuqtnKNTwiDPDlK3fNNQf3HCIzoN0d) trained for 50k steps on the new data. To use this model for evaluation/inference refer to [Training & Evaluation](#training--evaluation).

## Benchmarks

Multilingual model scores on test sets are given below. We are also releasing the [model-generated outputs](https://docs.google.com/uc?export=download&id=16OaTryW-QCaYevQ9Gtbr7aKhCJKBDEAr) for future analysis.

Language | ROUGE-1 / ROUGE-2 / ROUGE-L
---------|----------------------------
Amharic | 20.0485 / 7.4111 / 18.0753
Arabic | 34.9107 / 14.7937 / 29.1623
Azerbaijani | 21.4227 / 9.5214 / 19.3331
Bengali | 29.5653 / 12.1095 / 25.1315
Burmese | 15.9626 / 5.1477 / 14.1819
Chinese (Simplified) | 39.4071 / 17.7913 / 33.406
Chinese (Traditional) | 37.1866 / 17.1432 / 31.6184
English | 37.601 / 15.1536 / 29.8817
French | 35.3398 / 16.1739 / 28.2041
Gujarati | 21.9619 / 7.7417 / 19.86
Hausa | 39.4375 / 17.6786 / 31.6667
Hindi | 38.5882 / 16.8802 / 32.0132
Igbo | 31.6148 / 10.1605 / 24.5309
Indonesian | 37.0049 / 17.0181 / 30.7561
Japanese | 48.1544 / 23.8482 / 37.3636
Kirundi | 31.9907 / 14.3685 / 25.8305
Korean | 23.6745 / 11.4478 / 22.3619
Kyrgyz | 18.3751 / 7.9608 / 16.5033
Marathi | 22.0141 / 9.5439 / 19.9208
Nepali | 26.6547 / 10.2479 / 24.2847
Oromo | 18.7025 / 6.1694 / 16.1862
Pashto | 38.4743 / 15.5475 / 31.9065
Persian | 36.9425 / 16.1934 / 30.0701
Pidgin | 37.9574 / 15.1234 / 29.872
Portuguese | 37.1676 / 15.9022 / 28.5586
Punjabi | 30.6973 / 12.2058 / 25.515
Russian | 32.2164 / 13.6386 / 26.1689
Scottish Gaelic | 29.0231 / 10.9893 / 22.8814
Serbian (Cyrillic) | 23.7841 / 7.9816 / 20.1379
Serbian (Latin) | 21.6443 / 6.6573 / 18.2336
Sinhala | 27.2901 / 13.3815 / 23.4699
Somali | 31.5563 / 11.5818 / 24.2232
Spanish | 31.5071 / 11.8767 / 24.0746
Swahili | 37.6673 / 17.8534 / 30.9146
Tamil | 24.3326 / 11.0553 / 22.0741
Telugu | 19.8571 / 7.0337 / 17.6101
Thai | 37.3951 / 17.275 / 28.8796
Tigrinya | 25.321 / 8.0157 / 21.1729
Turkish | 32.9304 / 15.5709 / 29.2622
Ukrainian | 23.9908 / 10.1431 / 20.9199
Urdu | 39.5579 / 18.3733 / 32.8442
Uzbek | 16.8281 / 6.3406 / 15.4055
Vietnamese | 32.8826 / 16.2247 / 26.0844
Welsh | 32.6599 / 11.596 / 26.1164
Yoruba | 31.6595 / 11.6599 / 25.0898

## Multilingual ROUGE
  * See [rouge module.](multilingual_rouge_scoring/)
  
## Training & Evaluation
  * See [training and evaluation module.](seq2seq/)

## License
Contents of this repository are restricted to only non-commercial research purposes under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). Copyright of the dataset contents belongs to the original copyright holders.

## Citation
If you use any of the datasets, models or code modules, please cite the following paper:
```
@inproceedings{hasan-etal-2021-xlsum,
    title = "XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages",
    author = "Hasan, Tahmid  and
      Bhattacharjee, Abhik  and
      Islam, Md Saiful and
      Samin, Kazi  and
      Li, Yuan-Fang and
      Kang, Yong-Bin and 
      Rahman, M. Sohel  and
      Shahriyar, Rifat",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2021",
    month = "August",
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/2106.13822"
}
```
