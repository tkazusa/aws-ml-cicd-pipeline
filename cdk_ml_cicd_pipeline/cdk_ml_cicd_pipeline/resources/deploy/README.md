## CDKで環境を作成した後に必要な作業

`auto-ml-cicd-edge-deploy/edge_deploy/README.md` を参照してください

## Greengrassのセットアップ

コンポーネントのデプロイを行う場合は、事前にデプロイ先のGreengrassを用意しておく必要があります。
また、Greengrassにコンポーネントをデプロイする場合には、TESで利用されるRole似権限が付与されている必要があります。
ここでは、CDKで環境をセットアップした後に、コンポーネントのデプロイをする前に必要な手順を紹介します。

### Greengrassを動かす環境を準備

この例ではCloud9(Ubuntu Server 18.04 LTS)を利用しますが、Raspberry PiなどGreengrassが動作するデバイスを用意していただいても構いません。その場合は、「Dockerの実行環境を用意」の手順に進んでください。

- AWSのマネージメントコンソールよりCloud9のenvironmentを作成します。
  - CDKをデプロイしたのと同じリージョンを利用してください
  - https://docs.aws.amazon.com/cloud9/latest/user-guide/tutorials-basic.html

#### Dockerの実行環境を用意

デバイス上でデモ用のコンポーネントを実行する場合、Dockerの実行環境がセットアップされている必要があります。詳しいセットアップ方法は以下のページを参考に進めてください。
(Cloud9を環境として利用する場合は、この手順は不要です)

https://docs.aws.amazon.com/greengrass/v2/developerguide/run-docker-container.html

### Greengrassが利用するRoleの準備

この手順ではAWS CLIを利用してRoleの作成を行います。この作業は、AWS CLIがセットアップされている環境で実行してください(この作業はGreengrassと同じ環境である必要はありません)。
https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html

CLIの実行にはAWSのクレデンシャルが必要になりますので用意してください。

環境変数にリージョンを指定します。

```
export AWS_DEFAULT_REGION=
```

Policyの準備

``` 
cat << EOF > tes-role-policy.json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "credentials.iot.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
```

Roleの作成

```
GG_TES_ROLE_NAME=MLOps_EdgeDeploy_TES_Role

aws iam create-role \
--role-name ${GG_TES_ROLE_NAME} \
--assume-role-policy-document file://tes-role-policy.json
```

Policy ドキュメントの作成
 
このPolicyドキュメントは、Greengrassに必要な最低限の権限とコンポーネントのデプロイに必要なEDRの権限を付与します

```
cat << EOF > device-role-access-policy.json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer"
      ],
      "Resource": [
        "*"
      ],
      "Effect": "Allow"
    }
  ]
}
EOF
```

Policyの作成

```
TES_POLICY_ARN=$(aws iam create-policy \
--policy-name ${GG_TES_ROLE_NAME}_Policy \
--policy-document file://device-role-access-policy.json \
--query Policy.Arn \
--output text)

echo ${TES_POLICY_ARN}
```

PolicyをRoleに紐付け

```
aws iam attach-role-policy \
--role-name ${GG_TES_ROLE_NAME} \
--policy-arn ${TES_POLICY_ARN}
```

Role Aliasの作成

TESは、Aliasを指定して、Roleの権限を取得します

```
TES_ROLE_ARN=$(aws iam get-role \
--role-name ${GG_TES_ROLE_NAME} \
--query Role.Arn \
--output text)

echo ${TES_ROLE_ARN}

TES_ROLE_ALIAS_NAME=${GG_TES_ROLE_NAME}_Alias

echo ${TES_ROLE_ALIAS_NAME}

aws iot create-role-alias \
--role-alias ${TES_ROLE_ALIAS_NAME} \
--role-arn ${TES_ROLE_ARN}
```

実行結果

```
{
    "roleAlias": "MLOps_EdgeDeploy_TES_Role_Alias",
    "roleAliasArn": "arn:aws:iot:us-east-1:1234567890:rolealias/MLOps_EdgeDeploy_TES_Role_Alias"
}
```

コマンドを実行して表示された結果に含まれている"roleAlias" と、GG_TES_ROLE_NAMEに設定されている値を、Greengrassのセットアップ時に利用します。

echo ${GG_TES_ROLE_NAME} 



### Greengrassのインストール

この作業は、Greengrassをインストールするデバイス上(Cloud9または、ご自身で用意したデバイス)で実行します。

#### 環境変数の設定

Greengrassのセットアップに必要なため、ここでクレデンシャルを環境変数に設定します。リージョンは咲くほど作成したRoleと同じリージョンを指定します。
(セットアップ後はこのクレデンシャル情報は不要となります)

```
export AWS_DEFAULT_REGION=ap-northeast-1
export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
```

##### Greengrassで使う名前を変数に指定

Greengrassをセットアップする際にいくつか名前を指定する必要がありますので、変数に指定します。

- GG_THING_NAME
  - GreengrassのThing名を指定します。
- GG_THING_GROUP_NAME
  - Greengrassが所属するグループを指定します。同じ機能を持つGreengrassの場合、同じグループに含めることで、効率的にデプロイや設定変更を行うことができます。
- GG_TES_ROLE_NAME
  - GreengrassのTESで利用するロール名を指定します。先の手順で
- TES_ROLE_ALIAS_NAME
  - GreengrassのTESで利用するロールAlias名を指定します

設定例

```
GG_THING_NAME=MLOpsDevice
GG_THING_GROUP_NAME=MLOps_things_group
GG_TES_ROLE_NAME=MLOps_EdgeDeploy_TES_Role
TES_ROLE_ALIAS_NAME=MLOps_EdgeDeploy_TES_Role_Alias
```

#### Greengrassのセットアップ

Greengrassソフトウエアのダウンロード

```
curl -s https://d2s8p88vqu9w66.cloudfront.net/releases/greengrass-nucleus-latest.zip > greengrass-nucleus-latest.zip

unzip greengrass-nucleus-latest.zip -d GreengrassCore && rm greengrass-nucleus-latest.zip
```

Javaのインストール

V2からはJava 8以上が必要です。インストールされていない場合は、インストールしてください。

```
java --version
```

インストール

```
sudo apt install openjdk-11-jdk
```

セットアップの実行

このコマンドで、Thing、IoT Policy、証明書、GreengrassCoreが作成されます。

```
sudo -E java -Droot="/greengrass/v2" -Dlog.store=FILE \
  -jar ./GreengrassCore/lib/Greengrass.jar \
  --aws-region ${AWS_DEFAULT_REGION} \
  --thing-name ${GG_THING_NAME} \
  --thing-group-name ${GG_THING_GROUP_NAME} \
  --tes-role-name ${GG_TES_ROLE_NAME} \
  --tes-role-alias-name ${TES_ROLE_ALIAS_NAME} \
  --component-default-user ggc_user:ggc_group \
  --provision true \
  --setup-system-service true
```

#### インストールの動作確認

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