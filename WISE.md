## WISE

WISE is a benchmark dataset of Conversational Information Seeking (CIS). It contains **1.9K** conversation with many search intents, and **37K** utterances with an average turn number of **9.2**.

## Data

The conversation files include `conversation_train_line.json/conversation_test_line.json/conversation_valid_line.json`.
`conversation_test_line.json` is further divided into `conversation_testseen_line.json` and `conversation_testunseen_line.json`. 
The knowledge base file is `document_line.json`.

The format of conversation files is as follows:

```json
{
    "conversation_id": 11201,
    "conversations": [
        ...
        {
            "msg_id": 11278,
            "turn_id": 2,
            "role": "user",
            "intent": [
                "Reveal"
            ],
            "response": "我想了解一些现代的舞蹈"
        },
        {
            "msg_id": 11430,
            "turn_id": 2,
            "role": "system",
            "state": [
                "现代舞蹈"
            ],
            "query_candidate": [
                "超简单现代舞",
                "以后的以后舞蹈",
                "适合零基础女生跳的舞",
                "现代舞蹈教学",
                "现代舞蹈的特点",
                "现代舞蹈造型单人"
            ],
            "action": [
                "Clarify",
                "Choice"
            ],
            "selected_query": [
                "现代舞蹈教学",
                "现代舞蹈的特点"
            ],
            "response": "请问，您是想了解现代舞蹈的特点还是教学？",
            "document_candidate": [
                3212,
                3213,
                ...
                3227,
                3229
            ],
            "selected_document": [],
            "passage_candidate": [
                "3212-1",
                "3212-2",
                ...
                "3229-4",
                "3229-5"
            ],
            "selected_passage": []
        },
        {
            "msg_id": 11485,
            "turn_id": 3,
            "role": "user",
            "intent": [
                "Reveal"
            ],
            "response": "我想了解爵士舞"
        },
        {
            "msg_id": 11763,
            "turn_id": 4,
            "role": "system",
            "state": [
                "爵士舞",
                "基本信息"
            ],
            "query_candidate": [
                "查个人信息",
                ...
                "招工信息附近"
            ],
            "action": [
                "Answer",
                "open-text"
            ],
            "selected_query": [],
            "response": "它是美国现代舞，是一种急促又富动感的节奏型舞蹈，是属于一种外放性的舞蹈。",
            "document_candidate": [
                3251,
                3252,
                ...
                3262,
                3264
            ],
            "selected_document": [
                3251
            ],
            "passage_candidate": [
                "3251-1",
                "3251-2",
                ...
                "3264-18",
                "3264-19"
            ],
            "selected_passage": [
                "3251-8",
                "3251-13",
                "3251-20"
            ]
        }
        ...
    ],
    "intent": "intent:你想了解现代的一些舞蹈 description:首先查找爵士舞等现代舞种，给出形成时间，舞蹈人数等。然后找出能够欣赏现代舞种的网站，给出视频链接。之后给出学习舞蹈的机构所在地和网站，联系电话等。"
}
```

- `conversation_id` is the unique id of a conversation
- `conversations` is contents of conversation
  - `msg_id` is the unique id of a utterance
  - `turn_id` is the turn of a utterance in a conversation
  - `role` is the role of speaker who give current utterance
  - `intent` is intent of user's utterance
  - `state` is keyphrase of current utterance
  - `query_candidate` is a set of queries
  - `action` is the action of a utterance 
  - `selected_query` is selected queries from a set of candidate queries
  - `response` is the utterance the speaker gives
  - `document_candidate` is a set of candidate documents' ids
  - `selected_document` is selected documents' ids from a set of candidate documents
  - `passage_candidate` is a set of candidate passages' ids
  - `selected_passage` is selected passages' ids from a set of candidate passages
- `intent` is the topic of a conversation

And the format of documents of WISE dataset looks like the following:

```json
{
    "id": 46,
    "origin_link": "http://www.baidu.com/link?url=P214ZrP7DP_5wwcqE5pcYqiGtWFUGPFlpv11KfqKHlcC886P_MKP36qoAbo8knmjsTvOpUA09OyLWF9Yhm0vHdLumr7gI7GCIyISbasXdg7",
    "title": "环境保护法常识:突发环境污染事件的分类",
    "abstract": "2015年6月26日 - 突发环境污染事件不同于一般的环境污染,具有发生突然、扩散迅速、危害严重、污染物不明及处理的艰巨性等特点。 突发环境污染事件包括重点流域、敏感水...",
    "final_link": "http://www.chinalawedu.com/web/21676/wl1506261065.shtml",
    "passages": [
        ...
        {
            "id": 4,
            "sentences": [
                {
                    "id": 1,
                    "text": "突发环境污染事件包括重点流域、敏感水域水环境污染事件;重点城市光化学烟雾污染事件;危险化学品、废弃化学品污染事件;海上石油勘探开发溢油事件;突发船舶污染事件等。"
                },
                {
                    "id": 2,
                    "text": "辐射环境污染事件包括放射性同位素、放射源、辐射装置、放射性废物辐射污染事件。"
                }
            ]
        },
        ...
    ]
}
```

- `id` is the id of a document
- `origin_link` is the link of a document from the search engine
- `title` is the title of a document
- `abstract` is the abstract of a document
- `final_link` is the link of a document on its website
- `passages` is a set of passages of a document
  - `id` is the passage id of current document
  - `sentences` is a set of sentences of a passage
    - `id` is the sentence id of current passage
    - `text` is contents of current passage