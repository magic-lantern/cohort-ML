

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.bca704bf-dd62-41d9-81b8-bf01814f578f"),
    outcomes=Input(rid="ri.foundry.main.dataset.349f1404-e60e-4a76-9a32-13fe06198cc1")
)
SELECT count(1) as cnt, bad_outcome
FROM outcomes
group by bad_outcome

