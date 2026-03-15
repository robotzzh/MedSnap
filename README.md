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

Git 分支使用说明
1 查看当前分支

先查看当前所在分支：

git branch

当前分支前面会有 * 标记，例如：

* master
2 创建新的开发分支

从当前分支创建一个新的开发分支：

git checkout -b dev_XXX

例如：

git checkout -b dev_XXX

执行后会自动切换到该分支。

3 再次确认当前分支
git branch

输出示例：

  master
* dev_XXX

说明当前正在 dev_XXX 分支开发。

4 推送分支到 GitHub

第一次需要建立远程分支：

git push --set-upstream origin dev_XXX

例如：

git push --set-upstream origin dev_XXX

执行后：

GitHub 会创建 dev_XXX 分支

本地分支会自动关联远程分支

5 之后的提交

以后提交代码只需要：

git add .
git commit -m "your commit message"
git push