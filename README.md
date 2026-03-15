配置 GitHub SSH Key（新手教程）

如果你第一次使用 GitHub 推送代码，可能会遇到：

Permission denied (publickey)

这是因为 电脑还没有配置 SSH key。
按照下面步骤操作即可。

1 生成 SSH Key

打开 PowerShell / Terminal，输入：

ssh-keygen -t ed25519 -C "你的邮箱"

例如：

ssh-keygen -t ed25519 -C "example@email.com"

然后会看到提示：

Enter file in which to save the key

直接一直按 Enter（回车）即可。

生成成功后，系统会创建两个文件：

~/.ssh/id_ed25519
~/.ssh/id_ed25519.pub
2 复制 SSH 公钥

运行：

cat ~/.ssh/id_ed25519.pub

会输出一长串类似：

ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIxxxxxxxxxxxxxxxxxxxxxxxxxxx your_email

把这一整行复制下来。

3 添加到 GitHub

打开 GitHub：

https://github.com/settings/keys

点击：

New SSH key

填写：

Title: My Computer
Key: （粘贴刚才复制的内容）

然后点击 Add SSH key。

4 测试 SSH 是否成功

回到终端运行：

ssh -T git@github.com

如果看到：

Hi your-username! You've successfully authenticated

说明 SSH 配置成功。

5 推送代码

现在就可以正常推送代码了：

git push origin master

或者

git push origin main

# 切换分支
git checkout -b dev——XXX