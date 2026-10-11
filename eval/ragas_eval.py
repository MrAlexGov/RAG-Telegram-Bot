"""Оценка генерации через Ragas: даёт ли реранкинг выигрыш в качестве ответа.

Берёт contexts.json от retrieval_eval.py (top-3 чанка на вопрос для каждого
варианта), генерирует ответ по промпту бота и считает метрики Ragas:
faithfulness (ответ опирается на контекст), context_recall и
llm_context_precision_with_reference (насколько контекст покрывает эталон),
factual_correctness (сверка с эталонным ответом SberQuAD).

LLM для генерации и судья — любой OpenAI-совместимый endpoint
(EVAL_BASE_URL / EVAL_API_KEY / EVAL_MODEL). Мы использовали Kaggle Benchmarks
model proxy (Gemini 3 Flash), см. README.

Запуск: python eval/ragas_eval.py --n 40 --variants langchain,langchain+rerank
"""
import argparse, json, os, sys, time

from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall
from ragas.run_config import RunConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import PROMPT  # тот же промпт, что у бота


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="eval/results/contexts.json")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--variants", default="langchain,langchain+rerank,llamaindex,llamaindex+rerank")
    ap.add_argument("--out", default="eval/results/ragas.json")
    a = ap.parse_args()
    llm = ChatOpenAI(base_url=os.environ["EVAL_BASE_URL"], api_key=os.environ["EVAL_API_KEY"],
                     model=os.environ.get("EVAL_MODEL", "google/gemini-3-flash-preview"), temperature=0, timeout=120)
    judge = LangchainLLMWrapper(llm)
    data = json.load(open(a.contexts))
    qs = data["questions"][:a.n]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for v in a.variants.split(","):
        if v in res:
            continue
        rows = []
        for it, ctx in zip(qs, data["top3"][v][:a.n]):
            context = "\n\n".join(f"Источник: wiki (страница 0)\nТекст: {c}" for c in ctx)
            ans = None
            for attempt in range(8):  # шлюз/прокси иногда перезапускаются — ждём и повторяем
                try:
                    ans = llm.invoke(PROMPT.format(context=context, question=it["q"])).content
                    break
                except Exception as e:
                    print("gen retry", e); time.sleep(20)
            if ans is None:
                continue
            rows.append({"user_input": it["q"], "retrieved_contexts": ctx, "response": ans, "reference": it["answer"]})
        r = evaluate(EvaluationDataset.from_list(rows),
                     metrics=[Faithfulness(), LLMContextRecall(), LLMContextPrecisionWithReference(), FactualCorrectness(mode="recall")],
                     llm=judge, run_config=RunConfig(max_workers=4, timeout=240, max_retries=10, max_wait=90), show_progress=False)
        df = r.to_pandas()
        metric_cols = [m for m in df.columns if df[m].dtype.kind == "f"]
        res[v] = {m: round(float(df[m].mean()), 3) for m in metric_cols}
        res[v]["valid"] = {m: int(df[m].notna().sum()) for m in metric_cols}  # NaN = оценка упала, в среднее не входит
        res[v]["n"] = len(rows)
        print(v, res[v], flush=True)
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
