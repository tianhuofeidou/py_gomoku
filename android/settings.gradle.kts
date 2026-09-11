// 仓库顺序按环境切换：
//   本机：Maven Central 不可直连，走阿里云镜像；官方仓库只作兜底。
//   CI  ：GitHub Actions 会注入 CI=true，此时直连官方仓库，镜像只作兜底。
// 为什么必须切：阿里云镜像一旦返回 5xx，Gradle 会把该仓库整个禁用、依赖解析全灭
// （v1.5.0 那次 main 分支的 Android 构建就是这么挂的：502 Bad Gateway）。
pluginManagement {
    val officialFirst = System.getenv("CI") != null
    repositories {
        if (officialFirst) {
            google()
            gradlePluginPortal()
        }
        maven("https://maven.aliyun.com/repository/gradle-plugin")
        maven("https://maven.aliyun.com/repository/google")
        maven("https://maven.aliyun.com/repository/public")
        if (!officialFirst) {
            google()
            gradlePluginPortal()
        }
    }
}

dependencyResolutionManagement {
    val officialFirst = System.getenv("CI") != null
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        if (officialFirst) {
            google()
            mavenCentral()
        }
        maven("https://maven.aliyun.com/repository/google")
        maven("https://maven.aliyun.com/repository/public")
        if (!officialFirst) {
            google()
            mavenCentral()
        }
    }
}

rootProject.name = "Gomoku"
include(":app")
