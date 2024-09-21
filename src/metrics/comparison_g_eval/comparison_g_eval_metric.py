# Inspired originally by https://github.com/confident-ai/deepeval/blob/729d505c703cee93b70bf955ac0e261acf00288a/deepeval/metrics/g_eval/g_eval.py
"""LLM evaluated metric based on the GEval framework: https://arxiv.org/pdf/2303.16634.pdf"""

from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
    initialize_model,
    trimAndLoadJson,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
)
from deepeval.utils import get_or_create_event_loop, prettify_list

from metrics.comparison_g_eval.schema import ReasonScore, Steps
from metrics.comparison_g_eval.template import ComparisonGEvalTemplate


G_EVAL_PARAMS = {
    LLMTestCaseParams.INPUT: "Input",
    LLMTestCaseParams.ACTUAL_OUTPUT: "Actual Output",
    LLMTestCaseParams.EXPECTED_OUTPUT: "Expected Output",
    LLMTestCaseParams.CONTEXT: "Context",
    LLMTestCaseParams.RETRIEVAL_CONTEXT: "Retrieval Context",
}


def construct_comparison_g_eval_params_string(
    llm_test_case_params: list[LLMTestCaseParams],
):
    g_eval_params = [G_EVAL_PARAMS[param] for param in llm_test_case_params]

    if len(g_eval_params) == 1:
        g_eval_params_str = g_eval_params[0]
    elif len(g_eval_params) == 2:
        g_eval_params_str = " and ".join(g_eval_params)
    else:
        g_eval_params_str = ", ".join(
            g_eval_params[:-1]) + ", and " + g_eval_params[-1]

    return g_eval_params_str


class ComparisonGEval(BaseMetric):
    def __init__(
        self,
        name: str,
        evaluation_params: list[LLMTestCaseParams],
        criteria: str | None = None,
        evaluation_steps: list[str] | None = None,
        model: str | DeepEvalBaseLLM | None = None,
        threshold: float = 0.5,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        _include_g_eval_suffix: bool = True,
    ):
        self.name = name
        self.evaluation_params = evaluation_params

        # Check if both criteria and evaluation_steps are not None at the same time
        if criteria is None and evaluation_steps is None:
            raise ValueError(
                "Either 'criteria' or 'evaluation_steps' must be provided.")

        # Check if criteria is provided, it cannot be an empty string
        if criteria is not None and not criteria.strip():
            raise ValueError("Criteria provided cannot be an empty string.")

        # Check if evaluation_steps is provided, it cannot be an empty list
        if evaluation_steps is not None and len(evaluation_steps) == 0:
            raise ValueError(
                "'evaluation_steps' must not be an empty list. Either omit evaluation steps or include a non-empty list of steps."
            )

        self.criteria = criteria
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.evaluation_steps = evaluation_steps
        self.threshold = 1 if strict_mode else threshold
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self._include_g_eval_suffix = _include_g_eval_suffix

    def measure(self, test_case: LLMTestCase, _show_indicator: bool = True) -> float:
        check_llm_test_case_params(test_case, self.evaluation_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self, _show_indicator=_show_indicator):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(self.a_measure(
                    test_case, _show_indicator=False))
                return self.score
            else:  # noqa: RET505
                self.evaluation_steps: list[str] = self._generate_evaluation_steps(
                )
                score, reason = self.evaluate(test_case)
                self.reason = reason
                self.score = 0 if self.strict_mode and score < self.threshold else score
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Evaluation Steps:\n{prettify_list(self.evaluation_steps)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )

                return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
    ) -> float:
        check_llm_test_case_params(test_case, self.evaluation_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
        ):
            self.evaluation_steps: list[str] = await self._a_generate_evaluation_steps()
            score, reason = await self._a_evaluate(test_case)
            self.reason = reason
            self.score = 0 if self.strict_mode and score < self.threshold else score
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Criteria:\n{self.criteria}",
                    f"Evaluation Steps:\n{prettify_list(self.evaluation_steps)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )

            return self.score

    async def _a_generate_evaluation_steps(self) -> list[str]:
        if self.evaluation_steps:
            return self.evaluation_steps

        g_eval_params_str = construct_comparison_g_eval_params_string(
            self.evaluation_params)
        prompt = ComparisonGEvalTemplate.generate_evaluation_steps(
            criteria=self.criteria, parameters=g_eval_params_str
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["steps"]
        else:  # noqa: RET505
            try:
                res: Steps = await self.model.a_generate(prompt, schema=Steps)
                return res.steps
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["steps"]

    def _generate_evaluation_steps(self) -> list[str]:
        if self.evaluation_steps:
            return self.evaluation_steps

        g_eval_params_str = construct_comparison_g_eval_params_string(
            self.evaluation_params)
        prompt = ComparisonGEvalTemplate.generate_evaluation_steps(
            criteria=self.criteria, parameters=g_eval_params_str
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["steps"]
        else:  # noqa: RET505
            try:
                res: Steps = self.model.generate(prompt, schema=Steps)
                return res.steps
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["steps"]

    async def _a_evaluate(self, test_case: LLMTestCase) -> tuple[int | float, str]:
        text = """"""
        for param in self.evaluation_params:
            value = getattr(test_case, param.value)
            text += f"{G_EVAL_PARAMS[param]}:\n{value} \n\n"

        g_eval_params_str = construct_comparison_g_eval_params_string(
            self.evaluation_params)
        prompt = ComparisonGEvalTemplate.generate_evaluation_results(
            evaluation_steps=self.number_evaluation_steps(),
            text=text,
            parameters=g_eval_params_str,
        )

        try:
            # Don't have to check for using native model
            # since generate raw response only exist for deepeval's native model
            res, cost = await self.model.a_generate_raw_response(
                prompt, logprobs=True, top_logprobs=20
            )
            self.evaluation_cost += cost
            data = trimAndLoadJson(res.content, self)
            reason = data["reason"]
            score = ComparisonGEvalTemplate.calculate_score(data["choice"])

            return score, reason

        # This catches the case where a_generate_raw_response doesn't exist.
        except AttributeError:
            if self.using_native_model:
                res, cost = await self.model.a_generate(prompt)
                self.evaluation_cost += cost
                data = trimAndLoadJson(res, self)
                score = ComparisonGEvalTemplate.calculate_score(data["choice"])
                reason = data["reason"]
                return score, reason
            else:  # noqa: RET505
                try:
                    res: ReasonScore = await self.model.a_generate(prompt, schema=ReasonScore)
                    score = ComparisonGEvalTemplate.calculate_score(res.choice)
                    reason = res.reason
                    return score, reason
                except TypeError:
                    res = await self.model.a_generate(prompt)
                    data = trimAndLoadJson(res, self)
                    score = ComparisonGEvalTemplate.calculate_score(
                        data["choice"])
                    reason = data["reason"]
                    return score, reason

    def evaluate(self, test_case: LLMTestCase) -> tuple[int | float, str]:
        text = """"""
        for param in self.evaluation_params:
            value = getattr(test_case, param.value)
            text += f"{param.value}: {value} \n\n"

        g_eval_params_str = construct_comparison_g_eval_params_string(
            self.evaluation_params)

        prompt = ComparisonGEvalTemplate.generate_evaluation_results(
            evaluation_steps=self.number_evaluation_steps(),
            text=text,
            parameters=g_eval_params_str,
        )

        try:
            res, cost = self.model.generate_raw_response(
                prompt, logprobs=True, top_logprobs=20)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res.content, self)

            reason = data["reason"]
            score = ComparisonGEvalTemplate.calculate_score(data["choice"])

            return score, reason

        except AttributeError:
            # This catches the case where a_generate_raw_response doesn't exist.
            if self.using_native_model:
                res, cost = self.model.generate(prompt)
                self.evaluation_cost += cost
                data = trimAndLoadJson(res, self)
                return ComparisonGEvalTemplate.calculate_score(data["choice"]), data["reason"]
            else:  # noqa: RET505
                try:
                    res: ReasonScore = self.model.generate(
                        prompt, schema=ReasonScore)
                    score = ComparisonGEvalTemplate.calculate_score(res.choice)
                    reason = res.reason
                    return score, reason
                except TypeError:
                    res = self.model.generate(prompt)
                    data = trimAndLoadJson(res, self)
                    score = ComparisonGEvalTemplate.calculate_score(
                        data["choice"])
                    reason = data["reason"]
                    return score, reason

    def number_evaluation_steps(self):
        evaluation_steps = """"""
        for index, string in enumerate(self.evaluation_steps, start=1):
            evaluation_steps += f"{index}. {string}\n"
        return evaluation_steps

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:  # noqa: E722
                self.success = False
        return self.success

    @property
    def __name__(self):
        if self._include_g_eval_suffix:
            return f"{self.name} (ComparisonGEval)"
        else:  # noqa: RET505
            return self.name
