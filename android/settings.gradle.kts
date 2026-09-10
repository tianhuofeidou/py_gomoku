pluginManagement {
    repositories {
        // 国内镜像优先，官方仓库兜底。Maven Central 在本机不可直连，
        // 用阿里云 public 镜像代理；Google Maven 用阿里云 google 镜像。
        maven("https://maven.aliyun.com/repository/gradle-plugin")
        maven("https://maven.aliyun.com/repository/google")
        maven("https://maven.aliyun.com/repository/public")
        google()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        maven("https://maven.aliyun.com/repository/google")
        maven("https://maven.aliyun.com/repository/public")
        google()
    }
}

rootProject.name = "Gomoku"
include(":app")
