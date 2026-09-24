# 免费匿名部署

1. 使用专用匿名 GitHub 账号，创建私有仓库，上传本包解压后的文件。入口文件应位于仓库根目录。
2. 用已登录的 Streamlit Community Cloud 连接该 GitHub 账号，选择 Create app。
3. 选择仓库和分支，Main file path 填 `streamlit_app.py`，Python 版本选 **3.10**。
4. 在 Advanced settings → Secrets 粘贴部署者本地 `.streamlit/secrets.toml` 的内容。这个文件不在上传包中，不要提交到 GitHub。
5. 选择中性的应用子域名，部署后将应用访问权限设置为公开，再用未登录浏览器检查网页、菜单及分享页面是否包含身份信息。

网站免费托管；语言模型调用使用配置的 API 额度。访客可在侧栏切换自定义 Base URL、模型和 API Key，密钥仅用于自己的会话。默认密钥不显示在网页中。

代码和模型在云端运行，不连接内网 GPU。推荐模型资产由服务端从私有模型仓库下载，不通过前端暴露仓库地址或访问令牌。
