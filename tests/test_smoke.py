def test_authenticated_api_smoke_workflow(authenticated_api):
    client = authenticated_api["client"]
    headers = authenticated_api["headers"]

    assert client.get("/health/live").json() == {"status": "ok"}

    identity = client.get("/auth/me", headers=headers)
    assert identity.status_code == 200
    assert identity.json()["username"] == "test-owner"

    documents = client.get("/documents", headers=headers)
    assert documents.status_code == 200
    assert documents.json() == {"count": 0, "documents": []}

    assert client.get("/documents").status_code == 401
