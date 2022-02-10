## Componentのソースコードを管理するリポジトリにソースコードをPush

CDKの実行結果に表示されるOutputsの `ComponentCodeRepositoryURI` に出力された値を `<codecommit_uri>` に置き換えて実行します

```
cd 
git clone <codecommit_uri> greengrass_component_default_source
cd greengrass_component_default_source
cp ../auto-ml-cicd-edge-deploy/edge_deploy/component_source/* .
git add .
git commit -m "add source"
git push
```

## コンポーネントのコンテナイメージのベースをECRに登録
CodeBuildからGitHubの公開リポジトリのイメージを利用する場合、同一アドレスからのリクエスト制限に引っかかることがあるため、ベースとなるイメージを事前にECRに登録しておきます。
実際の運用では、さらにビルドに時間がかかるようなものも含めたイメージを作成しておくと、コンテナイメージのビルド時間を短縮させることが出来ます。

Outputsの `ComponentBaseImageRepositoryURI` に出力された値を `<ecr_uri>` に置き換えて実行します
この作業はh時期続きgit cloneした `greengrass_component_default_source` のディレクトリの中で行います。

```
REPO_URI=<ecr_uri>
REPO=`echo ${REPO_URI} | cut -d "/" -f 1`
REGION=`echo ${REPO_URI} | cut -d "." -f 4`

# docker imageをビルド
docker build -t inference-base -f Dockerfile_base .
docker tag inference-base:latest ${REPO_URI}:latest
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${REPO}
docker push ${REPO_URI}:latest
```