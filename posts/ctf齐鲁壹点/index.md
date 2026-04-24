# 记录一次逆向某小程序敏感信息与解密链路实战


## 前言

终于有一次 CTF 实操环节了！

上周六去围观了 ~[数据删除]~ 大会，看到了满墙的信息，顿时我想能不能把这些数据全爬下来？于是保存了一个二维码决定回去研究看看。

## 实战

### Fiddle 抓包尝试

打开二维码发现是个微信小程序，于是用 Fiddle 抓包，抓到了一个获取个人信息的接口：

![1](/image/CTF/qlyd/fiddle.png)

### 越权访问测试 (IDOR)

于是把这请求拉下来用 postman 请求一下，诶？居然拿到数据了，没有做任何的 token 校验！

然后把他的 `repositoryNo` 参数换一个人，发个请求，卧槽！还是能查到信息！

说明跟后面的 `encrypt` 参数也毫无关联。

### 加密数据观察

但是，查出来的信息手机号、QQ 号、照片 URL 是加密的，说个搞笑的，headImag 的值和 showPhotos 字段（加密）的明文是一样的。

![1](/image/CTF/qlyd/postman.png)

尝试解密手机号，经观察得出这可能是个 AES 加密，没有密钥解不出来。

### 逆向解包微信小程序

于是尝试逆向解包微信小程序。

众所周知，微信小程序在 `\Wechat File\WeChat Files\Applet` 这个目录下。但是我看了一下最后修改日期最近的怎么是 2023 年？感觉已经不是存在这里了。

于是打开 `Everything` 搜索 `Applet\` 目录，按时间排序，果然在
`C:\Users\XXX\AppData\Roaming\Tencent\xwechat\radium\users\649164f56d150de9876f6a1df91c8200\applet\packages` 这个路径下找到了可疑的小程序文件夹。

![1](/image/CTF/qlyd/applet.png)

首先按时间排序，你无法确定是哪个小程序、叫什么名字。所以这里我想了个办法：直接删除所有目录，重启微信，第一时间打开这个小程序，然后在此目录刷新一下，新生成的目录就是要解包的小程序。

一刷新，WTF，怎么有三个目录，不管了干脆全解包！

考虑使用工具：
[跨平台微信小程序反编译 GUI 工具，.wxapkg 文件扫描 &#43; 解密 &#43; 解包工具](https://github.com/wux1an/wxapkg)

解一下最新的前三个包。

![1](/image/CTF/qlyd/jiebao.png)

可以看到 3.3MB 的最可能是主程序，其他的是插件/组件。

### 解密逻辑溯源

观察目录结构，发现 `app-service.js` 存在大量逻辑。`Ctrl &#43; F` 搜索接口名，找到了请求方法：

```js
getInfo: function() {
            var t = this,
                e = c(16);
            i.GET({
                url: &#34;/gateway/appactivity/public/blindDate/repository/getInfo?repositoryNo=&#34;.concat(this.data.innerId, &#34;&amp;encrypt=&#34;).concat(encodeURIComponent(d(e)))
            }).then((function(n) {
                t.setData({
                    loading: !1
                });
                var a = n.code,
                    o = n.data;
                if (200 == a) {
                    var s = o;
                    s.tel = s.tel ? u(s.tel, e) : &#34;&#34;, s.phone = s.tel.replace(/(\d{3})\d{4}(\d{4})/, &#34;$1****$2&#34;), s.qq = s.qq ? u(s.qq, e) : &#34;&#34;, s.showPhotos = s.showPhotos ? u(s.showPhotos, e).split(&#34;,&#34;) : [], s.showVides = s.showVides ? u(s.showVides, e) : &#34;&#34;, console.log(&#34;userInfo&#34;, s), t.setData({
                        userInfo: s
                    }), t.getOhterInfo(), t.getVipData(), t.getRecommendData(), i.POST({
                        url: &#34;/gateway/app/activity/public/v2/blinddate/views&#34;,
                        data: {
                            innerId: s.innerId
                        }
                    })
                } else 301 == a &amp;&amp; (t.setData({
                    userInfo: {}
                }), wx.showToast({
                    title: &#34;二维码已经失效&#34;,
                    icon: &#34;none&#34;
                }))
            }))
        },
```

可以看出，`u()` 就是加密方法，传参部分有一个 `e=c(16)` 然后又 `d(e)`。我们考虑寻找一下这些函数。

找到一个定义方法：

```js
define(&#34;pages/active-details/active-details.js&#34;, function(require, module, exports, window, document, frames, self, location, navigator, localStorage, history, Caches, screen, alert, confirm, prompt, XMLHttpRequest, WebSocket, Reporter, webkit, WeixinJSCore) {
    &#34;use strict&#34;;
    var t, e = require(&#34;../../@babel/runtime/helpers/objectSpread2&#34;),
        n = require(&#34;../../@babel/runtime/helpers/defineProperty&#34;),
        a = getApp(),
        i = require(&#34;../../utils/https&#34;),
        o = require(&#34;../conversation/utils/utils&#34;),
        s = require(&#34;../../utils/payUp&#34;),
        r = require(&#34;../../utils/aes&#34;),
        c = r.generatekey,
        u = r.Decrypt,
        d = require(&#34;../../utils/rsa&#34;).rsaPublicData;
```

可以看出，他大概是采用 AES 和 RSA 混合加密。

于是我们需要重点关注 `r` 和 `d`，也就是要找到 `/utils/aes.js` 和 `/utils/rsa.js`

AES：

```js
define(&#34;utils/aes.js&#34;, function(require, module, exports, window, document, frames, self, location, navigator, localStorage, history, Caches, screen, alert, confirm, prompt, XMLHttpRequest, WebSocket, Reporter, webkit, WeixinJSCore) {
    &#34;use strict&#34;;
    var r = require(&#34;../miniprogram/miniprogram_npm/crypto-js/index&#34;);
    console.log(r);
    module.exports = {
        generatekey: function(r) {
            for (var e = &#34;ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789&#34;, n = &#34;&#34;, t = 0; t &lt; r; t&#43;&#43;) {
                var i = Math.floor(Math.random() * e.length);
                n &#43;= e.substring(i, i &#43; 1)
            }
            return n
        },
        Encrypt: function(e, n) {
            var t = r.enc.Utf8.parse(n),
                i = r.enc.Utf8.parse(e);
            return r.AES.encrypt(i, t, {
                mode: r.mode.CBC,
                iv: t,
                padding: r.pad.Pkcs7
            }).ciphertext.toString().toUpperCase()
        },
        Decrypt: function(e, n) {
            var t = r.enc.Utf8.parse(n),
                i = r.enc.Hex.parse(e),
                o = r.enc.Base64.stringify(i),
                a = r.AES.decrypt(o, t, {
                    iv: t,
                    mode: r.mode.CBC,
                    padding: r.pad.Pkcs7
                });
            return r.enc.Utf8.stringify(a).toString()
        }
    };
}, {
    isPage: false,
    isComponent: false,
    currentFile: &#39;utils/aes.js&#39;
});;
```
RSA：

```js
define(&#34;utils/rsa.js&#34;, function(require, module, exports, window, document, frames, self, location, navigator, localStorage, history, Caches, screen, alert, confirm, prompt, XMLHttpRequest, WebSocket, Reporter, webkit, WeixinJSCore) {
    &#34;use strict&#34;;
    Object.defineProperty(exports, &#34;__esModule&#34;, {
        value: !0
    }), exports.rsaJsonData = function(t) {
        var n = new e;
        return n.setPublicKey(r), n.encrypt(JSON.stringify(t))
    }, exports.rsaPrivateData = function(r) {
        var t = new e;
        return t.setPrivateKey(privateKey), t.encrypt(r)
    }, exports.rsaPublicData = function(t) {
        var n = new e;
        return n.setPublicKey(r), n.encrypt(t)
    };
    var e = require(&#34;./jsencrypt&#34;);
    console.log(232, e);
    var r = &#34;MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQCvT0vHJn/iebE3GIjng5efC67tZrRmBBMB1Y4RlLwP40kgtOh1LWTwp0aPlzuqBDFa2r1tq5AFl7TY1Ve3puAZhyuBIyF74uqlgXTysUtwdKysvD0AwPJyU6tKIYvfXIfaZjqdy&#43;0A743G2gscOHoSTfGj9hD/0mjEWbqo6mXblQIDAQAB&#34;;
}, {
    isPage: false,
    isComponent: false,
    currentFile: &#39;utils/rsa.js&#39;
});;
```

注意到 `c = r.generatekey`，然后根据他的 `generatekey` 函数逻辑，他就是生成一个 $16$ 位的随机数，然后把用 RSA 加密，其中 RSA的公钥直接拍你脸上了

```js
var r = &#34;MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQCvT0vHJn/iebE3GIjng5efC67tZrRmBBMB1Y4RlLwP40kgtOh1LWTwp0aPlzuqBDFa2r1tq5AFl7TY1Ve3puAZhyuBIyF74uqlgXTysUtwdKysvD0AwPJyU6tKIYvfXIfaZjqdy&#43;0A743G2gscOHoSTfGj9hD/0mjEWbqo6mXblQIDAQAB&#34;;
```

然后 AES 部分的密钥和便宜也完全用的是这个 16 位的随机数。

所以我们可以直接自己请求伪造一个 16 位的密钥，用他的 RSA 公钥加密，比如说我密钥就为 `1234567812345678`，使用 RSA 生成一个明文：

![1](/image/CTF/qlyd/rsa.png)

那么我只需要把这个
```Base64
YcXLwTDwqX&#43;cVXAO0g8dfvA6xLVnU8/w1IXfJ9wsjbr95WhqxBZSTkQtHxw/&#43;RXSb2SRCSn3c//AIbhCkaYbDak4eyZIiylXdCwZjvNQOLld8j&#43;FpuZbW1V/DbQboBhhq2n90Og8VyUzQdQHAtQ4MDhosYCGL626nE14wRILByA=
```
当作 `encrypt` 参数传进去就可以给后端伪造一个密钥。

用 URL 编码将此 Base64 串编一下发请求，果然，依旧可以请求成功：

![1](/image/CTF/qlyd/伪造请求.png)

并且我们拿到一个属于我们自己的 AES 明文，直接填入自己伪造的密钥进行解密：

![1](/image/CTF/qlyd/aes.png)

这里 `IV=Key`，是不安全的初始化向量实现。

解密成功！接下来就是跑脚本了。

## 防范措施

### 密钥生成权回归后端

密钥应由服务器动态生成并存储在 Session 或 Redis 中，与用户登录态绑定。避免由前端生成密钥上报，从根源消除“攻击者自定义钥匙”的逻辑漏洞。

### 后端数据脱敏

遵循最小必要原则。对于权限不足的用户，后端在查询数据库后应直接下发带掩码的脱敏数据（如 `138****1234`），确保即使加密逻辑被破解，攻击者也拿不到真实明文。

### 请求签名校验

引入基于时间戳、随机数及私有盐值（Salt）的签名机制。确保每一个 API 请求不可篡改且具有时效性，从而有效拦截脱离小程序环境的自动化抓取和重放攻击。


### 代码混淆与环境加固

利用小程序高级混淆功能对加解密逻辑进行函数名及流程混淆，并开启代码加密预览。这能极大提高逆向分析成本，防止攻击者轻易定位到 RSA 公钥和 AES 逻辑。

### 后端行为风控与限流

建立针对敏感接口的频率限制（Rate Limiting）。后端需实时监测异常的 ID 遍历行为（如短时间内请求大量不同 ID），并对可疑账号或 IP 执行封禁或验证码质询，防止全站数据被“拖库”。

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/ctf%E9%BD%90%E9%B2%81%E5%A3%B9%E7%82%B9/  

