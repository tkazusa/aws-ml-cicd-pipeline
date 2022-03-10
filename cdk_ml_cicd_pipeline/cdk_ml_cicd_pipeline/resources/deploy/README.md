# Greengrassの関連する設定

コンポーネントのデプロイを行う場合は、事前にデプロイ先のGreengrassを用意しておく必要があります。
また、Greengrassにコンポーネントをデプロイする場合には、TESで利用されるRole似権限が付与されている必要があります。
ここでは、CDKで環境をセットアップした後に、コンポーネントのデプロイをする前に必要な手順を紹介します。

## CDKで環境を作成した後に必要な作業

GreengrassのコンポーネントのもとになるスクリプトなどをCodeCommitやECRにアップロードしてください。具体的な手順は、[auto-ml-cicd-edge-deploy/edge_deploy/README.md](auto-ml-cicd-edge-deploy/edge_deploy/README.md) を参照してください

## Token Exchange Service用のRoleを作成

Greengrassが利用するTESのRoleとRole Aliasを作成します。

```
cd auto-ml-cicd-edge-deploy/edge_deploy/setup_tes_role.py
python setup_tes_role.py --device_name デバイス名 --region 利用リージョン
```

実行すると、Greengrassのインストールで実行するコマンドが表示されますので、メモしておきます。

## Greengrassを動かす環境の設定

この例ではCloud9(Ubuntu Server 18.04 LTS)を利用しますが、Raspberry PiなどGreengrassが動作するデバイスを用意していただいても構いません。その場合は、「Dockerの実行環境を用意」の手順に進んでください。

- AWSのマネージメントコンソールよりCloud9のenvironmentを作成します。
  - CDKをデプロイしたのと同じリージョンを利用してください
  - https://docs.aws.amazon.com/cloud9/latest/user-guide/tutorials-basic.html

### Dockerの実行環境を用意

デバイス上でデモ用のコンポーネントを実行する場合、Dockerの実行環境がセットアップされている必要があります。詳しいセットアップ方法は以下のページを参考に進めてください。
(Cloud9を環境として利用する場合は、この手順は不要です)

https://docs.aws.amazon.com/greengrass/v2/developerguide/run-docker-container.html


## Greengrassのインストール

この作業は、Greengrassをインストールするデバイス上(Cloud9または、ご自身で用意したデバイス)で実行します。

### 環境変数の設定

Greengrassのセットアップに必要なため、ここでクレデンシャルを環境変数に設定します。リージョンは咲くほど作成したRoleと同じリージョンを指定します。
(セットアップ後はこのクレデンシャル情報は不要となります)

```
export AWS_DEFAULT_REGION=ap-northeast-1
export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
```

### Javaのインストール

V2からはJava 8以上が必要です。インストールされていない場合は、インストールしてください。

```
java --version
```

インストール

```
sudo apt install openjdk-11-jdk
```

### Greengrassセットアップの実行

Greengrassソフトウエアのダウンロード

```
curl -s https://d2s8p88vqu9w66.cloudfront.net/releases/greengrass-nucleus-latest.zip > greengrass-nucleus-latest.zip

unzip greengrass-nucleus-latest.zip -d GreengrassCore && rm greengrass-nucleus-latest.zip
```

`Token Exchange Service用のRoleを作成` で表示された以下のようなコマンドを実行します。

```
sudo -E java -Droot="/greengrass/v2" -Dlog.store=FILE \
  -jar ./GreengrassCore/lib/Greengrass.jar \
  --aws-region AWS_DEFAULT_REGION \
  --thing-name GG_THING_NAME \
  --thing-group-name GG_THING_GROUP_NAME \
  --tes-role-name GG_TES_ROLE_NAME \
  --tes-role-alias-name TES_ROLE_ALIAS_NAME \
  --component-default-user ggc_user:ggc_group \
  --provision true \
  --setup-system-service true
```

### インストール後の動作確認

Greengrassをセットアップする際に `--setup-system-service true` を指定すると、サービスとして登録され、自動で起動します。
 
ステータスの確認

```
sudo systemctl status greengrass.service
```
以下のように正常に起動したログが出ていれば成功です。

```
● greengrass.service - Greengrass Core
   Loaded: loaded (/etc/systemd/system/greengrass.service; enabled; vendor preset: disabled)
   Active: active (running) since Sun 2021-12-26 06:32:14 UTC; 37s ago
 Main PID: 19801 (sh)
    Tasks: 41
   Memory: 91.0M
   CGroup: /system.slice/greengrass.service
           ├─19801 /bin/sh /greengrass/v2/alts/current/distro/bin/loader
           └─19835 java -Dlog.store=FILE -Dlog.store=FILE -Droot=/greengrass/v2 -jar /greengrass/v2/alts/current/distro/lib/Greengrass.jar --setup-system-service...

Dec 26 06:32:14 ip-172-31-48-156.ec2.internal systemd[1]: Started Greengrass Core.
Dec 26 06:32:14 ip-172-31-48-156.ec2.internal sh[19801]: Greengrass root: /greengrass/v2
Dec 26 06:32:14 ip-172-31-48-156.ec2.internal sh[19801]: JVM options: -Dlog.store=FILE -Droot=/greengrass/v2
Dec 26 06:32:14 ip-172-31-48-156.ec2.internal sh[19801]: Nucleus options: --setup-system-service false
Dec 26 06:32:19 ip-172-31-48-156.ec2.internal sh[19801]: Launching Nucleus...
Dec 26 06:32:21 ip-172-31-48-156.ec2.internal sh[19801]: Launched Nucleus successfully.
```

Greengrassのサービスを停止する場合

```
sudo systemctl stop greengrass.service
```
 
Greengrassのサービスを開始する場合

```
sudo systemctl start greengrass.service
```
 
ログの確認
 
Greengrassのログは、デフォルトだと `/greengrass/v2` に出力されます。

```
sudo tail -F /greengrass/v2/logs/greengrass.log
```