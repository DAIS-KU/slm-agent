# if decomp:
#     logger.info(f"Starting decomp(deocmp_mode {deocmp_mode})")
#     sub_tasks = decompose_task(
#         example=example,
#         augmented_question=augmented_question,
#         model_name=model_name,
#         key=key,
#         url=url,
#         model=model,
#         slm=slm,
#         inter_decomp=inter_decomp,
#         intra_inter_decomp=intra_inter_decomp,
#         retrieval_method=retrieval_method if decomp_ex else None,
#         top_k=3 if decomp_ex else None,
#         return_as_str=True,
#         multiple_decomp=multiple_decomp,
#     )
#     if action_planning:
#         action_plans = action_level_planning(
#             task=example["question"],
#             curruent_plans=sub_tasks,
#             model_name=model_name,
#             key=key,
#             url=url,
#             model=model,
#             slm=slm,
#             retrieval_method=retrieval_method if action_planning_ex else None,
#             top_k=3 if action_planning_ex else None,
#         )
#         additional_knowledge = action_plans
#     else:
#         additional_knowledge = sub_tasks
