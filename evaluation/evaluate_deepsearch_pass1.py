#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单轮评估脚本 (Pass@1)

功能：
1. 对单个预测文件进行评估
2. 使用 LLM 评判器判断答案正确性
3. 计算 Pass@1 准确率
4. 统计工具调用、token使用量、答案长度等
5. 生成 scored.jsonl 和 summary.jsonl
"""

from pydantic import BaseModel
from openai import OpenAI
import concurrent.futures
from typing import Literal
import litellm
import os
import argparse
import json
import concurrent
from tqdm import tqdm
from transformers import AutoTokenizer
import re
from prompt import *
import traceback
import tiktoken
import time
import threading
from zai import ZhipuAiClient
thread_local = threading.local()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY","")
os.environ['OPENAI_API_BASE'] = os.getenv("OPENAI_API_BASE","")
API_KEY= os.environ.get("SUMMARY_API_KEY", "6c4ffed3020c4af5bb82414ac8de9530.9RfQw9HQ4UdcC6ZS")
BASE_URL=os.getenv("BASE_URL","")

def get_client():
    if not hasattr(thread_local, 'client'):
        # thread_local.client = OpenAI(
        #     api_key=API_KEY,
        #     base_url=BASE_URL,
        # )
        # 使用BigModel模型
        thread_local.client = ZhipuAiClient(
            api_key=API_KEY)
    return thread_local.client

extracted_answer_format_for_confidence = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_answer",
        "schema": {
            "type": "object",
            "properties": {
                "extracted_final_answer": {"type": "string"},
                "reasoning": {"type": "string"},
                "correct": {"type": "string", "enum": ["yes", "no"]},
                "confidence": {"type": "number"},
                "strict": {"type": "boolean"},
            },
            "required": ["extracted_final_answer", "reasoning", "correct", "confidence", "strict"],
            "additionalProperties": False
        },
        "strict": True
    }
}

extracted_answer_format_for_xbench = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_answer",
        "schema": {
            "type": "object",
            "properties": {
                "最终答案": {"type": "string"},
                "解释": {"type": "string"},
                "结论": {"type": "string", "enum": ["正确", "错误"]},
            },
            "required": ["最终答案", "解释", "结论"],
            "additionalProperties": False
        },
        "strict": True
    }
}



def is_correct_judgement(judgement):
    return judgement.lower() == "correct" or (judgement and judgement.lower()[0] == "a")


def call_llm_judge(item):
    global judge_prompt, dataset, judge_model

    client = get_client()

    question = item["question"]
    correct_answer = item["answer"]
    response = item["prediction"].strip()
    prompt = judge_prompt.format(question=question, correct_answer=correct_answer, response=response)

    for attempt in range(100):
        try:
            if judge_model == "glm-4.5":
                response = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                )
                judgement = response.choices[0].message.content

            elif judge_model == "openai/qwen2.5-72b-instruct":
                response = litellm.completion(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    num_retries=5
                )
                judgement = response.choices[0].message["content"]
            elif judge_model == "google/gemini-2.0-flash-001":
                client = get_client()
                response_obj = client.beta.chat.completions.parse(
                    model=judge_model,
                    max_completion_tokens=8192,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    response_format=extracted_answer_format_for_xbench,
                    timeout=100.0
                )
                raw_judge = json.loads(response_obj.choices[0].message.content)
                judgement = "Correct" if raw_judge["结论"].lower() == "正确" else ""

            elif 'browsecomp' in dataset:
                os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY","")
                response = litellm.completion(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    num_retries=5,
                    response_format=extracted_answer_format_for_confidence
                )

                raw_content = response.choices[0].message["content"]
                raw_judge = json.loads(raw_content)
                judgement = "Correct" if raw_judge["correct"].lower() == "yes" else ""

            else:
                response = litellm.completion(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    num_retries=5
                )
                judgement = response.choices[0].message["content"]

            return {
                "question": question,
                "answer": correct_answer,
                "judgement": judgement
            }

        except Exception as e:
            if attempt == 4:
                print(f"Error judgement for question: {question}: {e}")
                return {
                    "question": question,
                    "answer": correct_answer,
                    "judgement": "Error",
                    "error": str(e)
                }
            time.sleep(3)
            continue


def process_single_round(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        items = [json.loads(line) for line in f]

    return items


def get_termination_value(item):
    if "termination" in item:
        return item["termination"]

    messages = item.get("messages", [])
    if not messages:
        return "unknown"

    last_message = messages[-1]["content"] if messages else ""


    if "max_turns_reached" in last_message.lower():
        return "max_turns_reached"
    elif "max_tokens_reached" in last_message.lower():
        return "max_tokens_reached"
    elif "<answer>" in last_message and "</answer>" in last_message:
        return "answered"
    else:
        return "unknown"


def count_tokens_with_tokenizer(text, tokenizer):
    try:
        if hasattr(tokenizer, 'encode'):
            return len(tokenizer.encode(text))
        else:
            return len(tokenizer.encode(text))
    except:

        return len(text) // 4


def single_round_statistics(input_file):
    contents = process_single_round(input_file)


    num_invalid, num_extra = 0, 0

    tool_use_cnt, visit_tool_cnt, search_tool_cnt, other_tool_cnt = [], [], [], []

    all_ans_lengths, all_think_lengths = [], []


    all_tool_calls_per_question = []
    all_assistant_tokens_per_question = []
    all_assistant_tokens_per_message = []
    termination_counts = {}

    try:
        tokenizer = AutoTokenizer.from_pretrained(os.getenv('TOKENIZER_MODEL_NAME', 'Qwen/Qwen2.5-7B-Instruct'))
    except Exception as e:
        tokenizer = tiktoken.encoding_for_model("gpt-4o")

    for item in contents:
        messages = item["messages"]
        final_msg = messages[-1]["content"] if len(messages) else ""


        if "<answer>" not in final_msg or "</answer>" not in final_msg:
            num_invalid += 1
            answer_length = 0
        else:
            answer_length = len(final_msg.split("<answer>")[1].split("</answer>")[0].strip())


        num_tool_use, num_visit_tool, num_search_tool, num_other_tool = 0, 0, 0, 0
        think_lengths = []
        question_assistant_tokens = 0


        for msg in messages:
            if msg['role'] == 'assistant':
                content = msg['content']


                remaining_content = content
                while "<tool_call>" in remaining_content and "</tool_call>" in remaining_content:
                    start_idx = remaining_content.find("<tool_call>")
                    end_idx = remaining_content.find("</tool_call>")
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        tool_call_content = remaining_content[start_idx + 11:end_idx].strip()
                        if tool_call_content:
                            num_tool_use += 1

                            try:
                                tool_call = json.loads(tool_call_content)
                                tool_name = tool_call.get('name', '')
                                if tool_name == 'search':
                                    num_search_tool += 1
                                elif 'visit' in tool_name:
                                    num_visit_tool += 1
                                else:
                                    num_other_tool += 1
                            except Exception:
                                if "visit" in tool_call_content:
                                    num_visit_tool += 1
                                elif "search" in tool_call_content:
                                    num_search_tool += 1
                                else:
                                    num_other_tool += 1

                        remaining_content = remaining_content[end_idx + 12:]
                    else:
                        break

                think_lengths.append(len(content))

                assistant_tokens = count_tokens_with_tokenizer(content, tokenizer)
                question_assistant_tokens += assistant_tokens
                all_assistant_tokens_per_message.append(assistant_tokens)

        tool_use_cnt.append(num_tool_use)
        visit_tool_cnt.append(num_visit_tool)
        search_tool_cnt.append(num_search_tool)
        other_tool_cnt.append(num_other_tool)

        all_ans_lengths.append(answer_length)
        think_length = sum(think_lengths) / len(think_lengths) if think_lengths else 0
        all_think_lengths.append(think_length)

        all_tool_calls_per_question.append(num_tool_use)
        all_assistant_tokens_per_question.append(question_assistant_tokens)

        termination = get_termination_value(item)
        termination_counts[termination] = termination_counts.get(termination, 0) + 1

        try:
            if len(tokenizer.encode("".join([msg["content"] for msg in messages]))) > 30000:
                num_extra += 1
        except:
            pass

    total_questions = len(contents)
    termination_freq = {k: round(v / total_questions, 3) for k, v in termination_counts.items()}

    return {
        "extra_length": num_extra,
        "num_invalid": num_invalid,
        "avg_action": sum(tool_use_cnt) / len(tool_use_cnt),
        "avg_visit_action": sum(visit_tool_cnt) / len(visit_tool_cnt),
        "avg_search_action": sum(search_tool_cnt) / len(search_tool_cnt),
        "avg_other_action": sum(other_tool_cnt) / len(other_tool_cnt),
        "avg_ans_length": sum(all_ans_lengths) / len(all_ans_lengths),
        "avg_think_length": sum(all_think_lengths) / len(all_think_lengths),
        "avg_tool_calls_per_question": sum(all_tool_calls_per_question) / len(all_tool_calls_per_question) if all_tool_calls_per_question else 0,
        "avg_assistant_tokens_per_question": sum(all_assistant_tokens_per_question) / len(all_assistant_tokens_per_question) if all_assistant_tokens_per_question else 0,
        "avg_assistant_tokens_per_message": sum(all_assistant_tokens_per_message) / len(all_assistant_tokens_per_message) if all_assistant_tokens_per_message else 0,
        "termination_freq": termination_freq
    }


def calculate_correct_statistics(results, items):
    """
    计算正确样本的统计信息（单轮版本）

    Args:
        results: 评估结果列表
        items: 原始数据项列表

    Returns:
        dict: 正确样本的统计信息
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(os.getenv('TOKENIZER_MODEL_NAME', 'Qwen/Qwen2.5-7B-Instruct'))
    except Exception as e:
        tokenizer = tiktoken.encoding_for_model("gpt-4o")

    correct_tool_calls = []
    correct_assistant_tokens = []

    for result in results:
        if not is_correct_judgement(result["judgement"]):
            continue

        try:
            matching_item = [item for item in items if item['messages'][1]['content'] == result['question']]
        except:
            items = [item for item in items if len(item['messages'])>0]
            matching_item = [item for item in items if item['messages'][1]['content'] == result['question']]

        if not matching_item:
            continue
        item = matching_item[0]

        messages = item["messages"]
        num_tool_use = 0
        question_assistant_tokens = 0

        for msg in messages:
            if msg['role'] == 'assistant':
                content = msg['content']

                # 正确统计工具调用次数
                remaining_content = content
                while "<tool_call>" in remaining_content and "</tool_call>" in remaining_content:
                    start_idx = remaining_content.find("<tool_call>")
                    end_idx = remaining_content.find("</tool_call>")
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        tool_call_content = remaining_content[start_idx + 11:end_idx].strip()
                        if tool_call_content:
                            num_tool_use += 1
                        remaining_content = remaining_content[end_idx + 12:]
                    else:
                        break

                # 统计完整的assistant token数
                assistant_tokens = count_tokens_with_tokenizer(content, tokenizer)
                question_assistant_tokens += assistant_tokens

        correct_tool_calls.append(num_tool_use)
        correct_assistant_tokens.append(question_assistant_tokens)

    avg_tool_calls_correct = sum(correct_tool_calls) / len(correct_tool_calls) if correct_tool_calls else 0
    avg_assistant_tokens_correct = sum(correct_assistant_tokens) / len(correct_assistant_tokens) if correct_assistant_tokens else 0

    return {
        "avg_tool_calls_per_question_correctly_solved": round(avg_tool_calls_correct, 3),
        "avg_assistant_tokens_per_question_correctly_solved": round(avg_assistant_tokens_correct, 3)
    }


def main():
    global judge_prompt, dataset, judge_model

    parser = argparse.ArgumentParser(description="Evaluate model predictions (Pass@1)")
    parser.add_argument("--input_file", required=True, help="Path to prediction file (e.g., iter1.jsonl)")
    parser.add_argument("--restore_result_path", default='summary_pass1.jsonl', help="Record result")
    parser.add_argument("--dataset", type=str, default="browsecomp_en",
                        choices=["gaia", "browsecomp_zh", "browsecomp_en_full", "webwalker", "xbench-deepsearch"])
    args = parser.parse_args()

    dataset = args.dataset
    if dataset in ["gaia", "webwalker"]:
        # judge_model = "openai/qwen2.5-72b-instruct"
        judge_model = "glm-4.5"
        judge_prompt = JUDGE_PROMPT_GAIA
    elif dataset in ["xbench-deepsearch"]:
        judge_prompt = JUDGE_PROMPT_XBENCH
        judge_model = "google/gemini-2.0-flash-001"
    elif dataset.startswith("browsecomp_zh"):
        judge_model = "gpt-4o-2024-08-06"
        judge_prompt = JUDGE_PROMPT_BROWSECOMP_OFFICIAL
    elif dataset.startswith("browsecomp_en"):
        judge_model = "gpt-4o-2024-08-06"
        judge_prompt = JUDGE_PROMPT_BROWSECOMP_OFFICIAL
    else:
        judge_model = "openai/qwen2.5-72b-instruct"
        judge_prompt = JUDGE_PROMPT_GAIA

    print(f"Using {dataset} judge prompt ...")
    print(f"Judge prompt:\n {judge_prompt}")
    print(f"Judge model:\n {judge_model}")

    # 检查文件是否存在
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    # 读取数据
    items = process_single_round(args.input_file)
    print(f"Loaded {len(items)} items from {args.input_file}")

    # 并行评估
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(call_llm_judge, item): item for item in items}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Evaluating"):
            results.append(future.result())

    # 生成 scored.jsonl
    scored_file = args.input_file.replace(".jsonl", "_scored.jsonl")
    sorted_results = sorted(results,
                          key=lambda x: items.index(next(item for item in items if item["question"] == x["question"])))

    with open(scored_file, 'w', encoding='utf-8') as f:
        for orig_item, scored_result in zip(items, sorted_results):
            scored_item = {
                "is_correct": is_correct_judgement(scored_result["judgement"]),
                "judgement": scored_result["judgement"]
            }
            if "error" in scored_result:
                scored_item["error"] = scored_result["error"]

            scored_item.update(orig_item)
            f.write(json.dumps(scored_item, ensure_ascii=False) + '\n')

    print(f"Saved scored results to: {scored_file}")

    # 计算准确率
    correct_count = sum(1 for r in results if is_correct_judgement(r["judgement"]))
    pass_at_1 = round(correct_count / len(results) * 100, 2)

    # 计算统计信息
    statistics = single_round_statistics(args.input_file)
    enhanced_statistics = calculate_correct_statistics(results, items)

    # 输出结果
    print(f"\n===========")
    print(f"Pass@1: {pass_at_1}%")
    print(f"Correct: {correct_count} / {len(results)}")
    print(f"\n# Invalid {statistics['num_invalid']}  # Extra Length {statistics['extra_length']}")
    print(f"Avg. Action {statistics['avg_action']:.2f}  Avg. Visit Action {statistics['avg_visit_action']:.2f}  Avg. Search Action {statistics['avg_search_action']:.2f}  Avg. Other Action {statistics['avg_other_action']:.2f}")
    print(f"Avg. Answer Length {statistics['avg_ans_length']:.2f}  Avg. Thinking Length {statistics['avg_think_length']:.2f}")
    print(f"\n=== ADDITIONAL STATISTICS ===")
    print(f"Avg. Tool Calls per Question: {statistics['avg_tool_calls_per_question']:.2f}")
    print(f"Avg. Tool Calls per Question (Correctly Solved): {enhanced_statistics['avg_tool_calls_per_question_correctly_solved']:.2f}")
    print(f"Avg. Assistant Tokens per Question: {statistics['avg_assistant_tokens_per_question']:.2f}")
    print(f"Avg. Assistant Tokens per Question (Correctly Solved): {enhanced_statistics['avg_assistant_tokens_per_question_correctly_solved']:.2f}")
    print(f"Avg. Assistant Tokens per Message: {statistics['avg_assistant_tokens_per_message']:.2f}")

    print(f"\n=== TERMINATION FREQUENCIES ===")
    for termination_type, frequency in statistics['termination_freq'].items():
        print(f"{termination_type}: {frequency:.3f}")

    print(f"===========\n")

    # 保存到 summary
    overall_eval_dict = {
        "dataset": dataset,
        "file": args.input_file,
        "pass_at_1": pass_at_1,
        "correct_count": correct_count,
        "total_count": len(results),
        "statistics": {**statistics, **enhanced_statistics}
    }

    with open(args.restore_result_path, 'a', encoding='utf-8') as jsonl_file:
        jsonl_file.write(json.dumps(overall_eval_dict, ensure_ascii=False) + '\n')

    print(f"Appended results to: {args.restore_result_path}")


if __name__ == "__main__":
    judge_prompt, dataset = None, ""
    try:
        main()
    except Exception as e:
        error_str = traceback.format_exc()
        print(f"Evaluation Failed: {e}")
        print("Trace Back", error_str)
