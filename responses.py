# responses = queue.Queue()  # Thread-safe queue to aggregate responses
#
# response = client.chat.completions.create(
#     model="gpt-4-turbo-2024-04-09",
#     response_format={"type": "json_object"},
#     messages=[
#         {"role": "user", "content": prompt}
#     ]
# )


# def worker(entity1, entity2):
#     match_response = make_initial_api_call(entity1, entity2)
#     updated_entity = update_entity_with_new_info(entity1, entity2, match_response)
#     if updated_entity:
#         responses.put(updated_entity)


# start_time = time.time()
#
# threads = []
#
# for entity1 in customers:
#     for entity2 in extended_customers:
#         t = threading.Thread(target=worker, args=(entity1, entity2))
#         t.start()
#         threads.append(t)
#
# for t in threads:
#     t.join()
#
# aggregated_responses = []
# while not responses.empty():
#     aggregated_responses.append(responses.get())
#
# pprint.pprint(aggregated_responses)
#
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Execution time: {execution_time} seconds")