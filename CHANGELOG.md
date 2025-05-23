# Changelog

## [0.4.0](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/compare/v0.3.0...v0.4.0) (2025-01-14)


### Features

* Update gecko model version to "text-embedding-005". ([#519](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/519)) ([0f23a65](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/0f23a65d9ce5556f0448d1e3f046aca8e5b0878a))

## [0.3.0](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/compare/v0.2.0...v0.3.0) (2024-12-03)


### Features

* Add similarity threshold to amenity search ([#477](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/477)) ([c49bef9](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/c49bef9901e81bee01088d975cdedbce5b89af8d))
* Add tracing for AlloyDB and CloudSQL Postgres providers ([#494](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/494)) ([2fa03bc](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/2fa03bcec54fd2fe4b463a54df772ae5e6490577))
* Consolidate postgres providers ([#493](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/493)) ([a3b2c42](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/a3b2c42735c5a0dfd58ce268637cce8253061ff3))
* Reuse connector object across different database connections in… ([#487](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/487)) ([61c0f52](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/61c0f5200a9853efa8097e1143ffc317b6f74777))
* Switch the llm to ChatVertexAI ([#486](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/486)) ([479c5e5](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/479c5e552394c473891389c2a7bac2a0ab3c75ce))


### Bug Fixes

* Add test cases to improve coverage for postgres and over more tools. ([#508](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/508)) ([20870ea](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/20870eafe7983863f8aeb41f1cca28d85583c16c))
* Reuse connector object across different database connections. ([#484](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/484)) ([2b05739](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/2b05739b0eb761738cc8db06d76f64ad7d199a2e)), closes [#416](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/416)
* update close client function to async ([#505](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/505)) ([b614276](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/b614276d7d746b24958fa1ccc067ea62170bd967))

## [0.2.0](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/compare/v0.1.0...v0.2.0) (2024-08-28)


### Features

* Add langgraph orchestration ([#447](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/447)) ([8cefed0](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/8cefed07c5e4cc4357d08fc3a29920dc2cfabd6a))
* add ticket validation and insertion to cloudsql postgres ([#437](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/437)) ([a4480fa](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/a4480fa1fd0c117d64278fa9d864647a96f9b8a8))
* Add tracing to langgraph orchestration and postgres provider ([#473](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/473)) ([a5759e9](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/a5759e995c86cfc310dbabcb29d4623d9172cdd3))
* Adding support for Spanner with PG Dialect in Database Retriever Service ([#469](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/469)) ([47ff11e](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/47ff11e734dec2ebc968846c522c3538e5a0eeeb))
* Implement llm system evaluation ([#440](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/440)) ([a2df60b](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/a2df60b4a36c9af1de3b9b56c4db62ef997535f4))
* Remove user ID and user email from `list_tickets()` result ([#464](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/464)) ([5958938](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/59589380e1f3c5382182fffb9a2ba9bc69f5a087))


### Bug Fixes

* update pytest to pytest_asyncio for async fixtures ([#474](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/474)) ([c2ad4bb](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/c2ad4bbd81fb49a262b165b2ffcdaea0b13c1c6f))
* update return from tools for langchain and function calling ([#476](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/476)) ([9dfb60b](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/9dfb60b2e9601d9631a4a1f004bcf63a107a9cb9))

## 0.1.0 (2024-07-01)


### Features

* Add "airports" dataset ([#8](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/8)) ([e566554](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/e566554ceda415a1d7d1a1a6bcacaa7b8fbf72ad))
* add "amenities" dataset ([#9](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/9)) ([74db3ff](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/74db3ff6b1a1fe00edf999e242c5f5587dec7a10))
* add "flights" dataset  ([#10](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/10)) ([eebd3ce](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/eebd3ce384c0443ab15d69d322c838bc472fb25f))
* add api endpoints for amenities ([#12](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/12)) ([bd1c411](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/bd1c41195c33c65440a8b84406be53c1173fd127))
* add app_test cicd workflow ([#157](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/157)) ([51f0e51](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/51f0e51e896cf2e29bfdcd9e14b52ccc4300be68))
* add clean up instructions ([#99](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/99)) ([c9021b1](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/c9021b17480f2c7c235fd241e86147fc6a49a1ef))
* add cloud sql cloudbuild workflow ([#143](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/143)) ([3dd3444](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/3dd3444e450316131b37a14b3a00722194921ec2))
* add Cloud SQL MySQL as a datastore provider ([#415](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/415)) ([d3f43d7](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/d3f43d78176d194402778ef650dde57650a5afbc))
* add cloudsql integration test ([#141](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/141)) ([764ffeb](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/764ffebd8fb0f7f8b86b09ef5308428c7a6aa749))
* add cloudsql provider ([#5](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/5)) ([32a063d](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/32a063df7736260608b04f23e170eac297affdfa))
* add confirmation for ticket booking ([#407](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/407)) ([6987280](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/69872805d55eb7332bf533417d44faf2f39b9687))
* add cymbal air passenger policy ([#265](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/265)) ([199a67c](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/199a67ca0d8c7d416f164f57642f0f7aa35849d8))
* add endpoint for policy ([#271](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/271)) ([397ca79](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/397ca79af0bd171998ca525e73deb13adebabc15))
* Add Firestore provider ([#65](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/65)) ([0cbd8ed](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/0cbd8eda29d19f2f547412320a120909ebcd9c2c))
* add GET endpoint for airport ([#11](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/11)) ([d36fd29](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/d36fd29a3517c1b326ebacc9d1a7966efb2527ed))
* add initial "flights" endpoints ([#17](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/17)) ([da374d9](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/da374d941c6fdacf27329d2448ad121d23b83dd9))
* add initial readme ([#31](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/31)) ([cb6d68c](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/cb6d68c16015f54f965f889d35335f044d8d5cb5))
* add instructions file for cloudsql_postgres ([#93](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/93)) ([ca3d2c5](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/ca3d2c59673581429c1b2aa9c6b29dc9ab9a246e))
* add instructions for alloydb ([#27](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/27)) ([a30d5d1](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/a30d5d127d2c18f400c9f5aea5764a7cc5fcd9a9))
* add integration test for postgres ([#123](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/123)) ([0aeab54](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/0aeab54c2f5f157536f80415acb06cb65f91fde2))
* add labels syncing config ([#40](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/40)) ([b27558f](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/b27558fc7bb96e2da7617775d0d196e158712c48))
* add mock for retrieval service test ([#126](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/126)) ([498c6d7](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/498c6d736c0efc7c11785e2af6cf802e196cb6d4))
* add orchestration interface ([#226](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/226)) ([573c04b](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/573c04ba9394de20d1598a22c97876234954209a))
* Add Renovate Config ([#42](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/42)) ([40c221b](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/40c221bc2c18ec8d4841ec355b0eb88494183e3d))
* add search airport tools to langchain agent ([#140](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/140)) ([94a5df3](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/94a5df33ac8aeea5e8a6c9f854c84dd550699a11))
* add smoke test for function calling ([#199](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/199)) ([5eecedf](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/5eecedf2678d8b03906ba934992069a8a8564268))
* add test for database export ([#133](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/133)) ([b114186](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/b1141861afa4c55de64449776cfc7e33e4932ffd))
* Adding Spanner database provider for retrieval service. ([#316](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/316)) ([8990bc9](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/8990bc9c5794c592f978adf2f7c04f79d03a5859))
* Create google sign in button and send id token with request ([#147](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/147)) ([c0a34a7](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/c0a34a77e0bc54d1ac535fc8be834c70b133ab07))
* Create individual user client session ([#137](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/137)) ([359e5d3](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/359e5d37b52c542fff0d475d92b85374c58ed044))
* Create reset button to clear session cookies ([#152](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/152)) ([f008c04](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/f008c04678ab93d1f9634eb5cb6b499a9970e63b))
* Create Tickets endpoints ([#160](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/160)) ([26ad5d9](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/26ad5d9e7fd6138c429648886e086ef289e7cada))
* **doc:** Add Firestore setup instruction ([#87](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/87)) ([e6e6fb6](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/e6e6fb6b1b02e0ade9d012e3ac3d741f2f1f9506))
* **docs:** add code of conduct ([#39](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/39)) ([0dee8dd](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/0dee8dde43c55b7f2312f47ed02933f6d5814bf7))
* **docs:** update docs ([#36](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/36)) ([e908e75](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/e908e75ea308de1690ecfdf2068373fd614539fb))
* **docs:** update gcloud track for deployment with VPC network  ([#80](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/80)) ([84313c9](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/84313c92c2856c411b6071d9644fc12baeda5ce4))
* implement signin signout ([#258](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/258)) ([88b6c83](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/88b6c83b5c83d1ff0957ed5b5daeef9e1f04e905))
* increase font-size and make UI better ([#292](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/292)) ([c6dd0f6](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/c6dd0f63bdfd0a6113b68c35aae1d956ad664678))
* inital prototype of extension ([a502f84](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/a502f8412c15e1190d0317437a85db6df6f3d3d0))
* minor UI improvements ([#275](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/275)) ([7e31ac3](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/7e31ac36dde6359230bc168dce00a6e0ce72d56b))
* minor UI updates ([#279](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/279)) ([774703a](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/774703a14b0f850d8a6a4955946b2b87d17c680a))
* refactor demo frontend ([#15](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/15)) ([4245020](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/4245020a7548d61b6e76eec21d7a798672a12b08))
* refresh UI ([#239](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/239)) ([0254ce8](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/0254ce8adc412c3db4cd19560ff537d8cb615957))
* replace October flights data ([#37](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/37)) ([c2c70a2](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/c2c70a2bb08e0987a2cd56dd51feb723c950f9cb))
* Replace requests package with aiohttp ([#125](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/125)) ([4fc0d27](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/4fc0d279e5c5a49a468bea2bdf6d8b5cce158d26))
* update airport dataset and add search endpoint ([#14](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/14)) ([baabfe4](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/baabfe47f5b4345b441fefdf96524363dae0cd31))
* update amenity embedding to include amenity name ([#144](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/144)) ([3fe22ce](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/3fe22ce6e4950462adec34dfa80a73aa272e6f7a))
* update amenity_dataset with new hour columns ([#151](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/151)) ([160d8df](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/160d8df659bad2a8a56d9ed91121c7c9c8e63efe))
* update app in langchain demo ([#225](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/225)) ([7c463d9](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/7c463d91088a2d230658f58db9ac627fd09f2b49))
* update model to Gemini ([#158](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/158)) ([09959fd](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/09959fde044d82503b39bf1059ef4b67c6028471))
* Update prompt and tools ([#34](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/34)) ([499f6a7](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/499f6a7c6f1cfd027c5bd36a2792a3af0ba85b6f))
* update service Dockerfile ([#19](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/19)) ([ea7edc7](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/ea7edc7c18ca935f383cf177e13a56db3f796195))
* update to include user name in chat ([#249](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/249)) ([070b7e0](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/070b7e03315eb44caae5e19156e9d9ea9529e251))
* vertex ai function calling llm ([#188](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/188)) ([ab1aedc](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/ab1aedc0a1fa9e09a0efe080425344ddbb39ef41))


### Bug Fixes

* add dependency ([#195](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/195)) ([65cc37c](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/65cc37c4aa15e4b2ff00617936472f708a15c5ff))
* add retrieval service service account role to doc ([#121](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/121)) ([8ea12c9](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/8ea12c98f86d19832291cbe45e2bb5c9b044fd91))
* **ci:** use script over args in all cloudbuild ([#307](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/307)) ([da8d3f6](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/da8d3f6df691cb4c87872f42f95b63f772686904))
* **ci:** use script over args in cloudbuild syntax ([#306](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/306)) ([b5aa667](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/b5aa6670fdfe55b22a832ddb7df2248966ccde0f))
* Do not pass None values to session.get() ([#136](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/136)) ([5d9cbbc](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/5d9cbbc1e2c673d7b81b4ebd53a71d841fe4f5eb))
* **docs:** fixed broken links to demo apps ([#132](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/132)) ([4955cd0](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/4955cd09cf679756174739197d430f7cd5edfdb5))
* **firestore:** Add ID to all documents in Firestore provider ([#94](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/94)) ([1a02328](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/1a023289239c8b8625e27c0121f5cfc80875d642))
* Fix closed connection issue on reset button ([#175](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/175)) ([21607ec](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/21607ec76d7a9523ae3b4f63d9e991b0f287526a))
* Fix get_agent() uuid comparison ([#200](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/200)) ([bc68633](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/bc686338384cc23fcb15ee2f99097a962a578f51))
* fix the refresh icon hiding behind Google banner ([#247](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/247)) ([1d858ba](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/1d858baeccfd0b6157bf5e98bc793e8e7c5aeeb1))
* **langchain_tools_demo:** Add ID Token credential flow for GCE ([#198](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/198)) ([ed9b6c2](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/ed9b6c2ac7e076f2e220c2336917c8541f0f485d))
* **langchain_tools_demo:** fix agent concurrency between restarts ([#194](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/194)) ([2584154](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/258415450828ee602f36a5f52b25942d06f4a1c5))
* **langchain_tools_demo:** use relative paths for resources ([#192](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/192)) ([6f8ae0d](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/6f8ae0dec6f34276df304ea67acfeea77d979f65))
* Make `clientId `optional in config.yml ([#207](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/207)) ([3914d8c](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/3914d8c105ab117295da9e6343480c42489e24a8))
* missing Client ID ([#196](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/196)) ([960f6b1](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/960f6b1476f0579c80bb400fbb808d0c8a1348ab))
* sign out when user token invalid ([#329](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/329)) ([2ec915b](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/2ec915b5a9c0902c47e30699b5eb3acade007792))
* strip query params in loginUri ([#425](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/425)) ([41f3bb7](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/41f3bb7c97415afea4e6a0e53b828055880d9990))
* update AlloyDB instruction order ([#92](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/92)) ([334257c](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/334257cc4475f5330a3b5748c456bc1ec06390f5))
* update alloydb.md to remove extra trailing space ([#46](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/46)) ([7f97d33](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/7f97d33d222e9180108affed20515f4b37745eab))
* update embeddings and pin model ([#124](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/124)) ([5842c08](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/5842c08a3864dabc12ebd03e4fe39aa912868c11))
* update renovate config ([#378](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/378)) ([2ec52ce](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/2ec52cee3b0afbde59e60fdad87aa8d8211a0e8c))
* update search airport tool ([#148](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/148)) ([d4b36e9](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/d4b36e93f04b6c3c3df57f6c5ffcc1c3554d7723))
* Use idiomatic python for conditional ([#149](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/149)) ([18e5e9d](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/18e5e9d0c58b83fc6a0af6292a722fe0e74f4527))
* use lazy refresh for AlloyDB and Cloud SQL Connector ([#429](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/429)) ([c73484d](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/c73484d60b2724eb096ecd6a568c1747edfd5e26))


### Miscellaneous Chores

* release 0.1.0 ([#431](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/issues/431)) ([0f378de](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app/commit/0f378de0fc441bca33c908c88b5678344303e64f))
