

@transform_pandas(
    Output(rid="ri.vector.main.execute.2a07a055-a516-4591-befb-fe274141f319"),
    outcomes=Input(rid="ri.foundry.main.dataset.349f1404-e60e-4a76-9a32-13fe06198cc1")
)
SELECT *
FROM outcomes

