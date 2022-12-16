import arxiv

search = arxiv.Search(query = "Gravitational wave",  id_list= [],
  max_results = float('inf'),
  sort_by = arxiv.SortCriterion.Relevance,
  sort_order = arxiv.SortOrder.Descending
)

index = 0
with open('./data/GWtext/GWAbstract.raw', 'w') as f:
  for i in search.results():
    f.write(i.summary.replace("\n"," ")+"\n")
    index += 1
    if index%100 == 0: print(index)

