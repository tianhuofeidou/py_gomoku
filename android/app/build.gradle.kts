plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.compose.compiler)
  alias(libs.plugins.chaquopy)
}

// CI 可用 -PversionName / -PversionCode 覆盖；本地默认 v1.5.0。
// versionCode 规则：major * 10000 + minor * 100 + patch。
val appVersionCode = providers.gradleProperty("versionCode").orNull?.toIntOrNull() ?: 10500
val appVersionName = providers.gradleProperty("versionName").orNull ?: "1.5.0"

// CI 必须通过环境变量指定 Python 3.14；本地默认 D 盘路径。
val chaquopyBuildPython = System.getenv("CHAQUOPY_BUILD_PYTHON")
    ?: "D:/python3.14/python.exe"

android {
    namespace = "com.example.gomoku"
    compileSdk = 36
    defaultConfig {
        applicationId = "com.example.gomoku"
        minSdk = 24
        targetSdk = 36
        versionCode = appVersionCode
        versionName = appVersionName

        ndk {
            // 引擎是纯 Python，不依赖第三方包；Python 3.12+ 仅支持 64 位 ABI。
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
      compose = true
      aidl = false
      buildConfig = true
      shaders = false
    }

    packaging {
      resources {
        excludes += "/META-INF/{AL2.0,LGPL2.1}"
      }
    }
}

kotlin {
    jvmToolchain(17)
}

chaquopy {
    defaultConfig {
        version = "3.14"
        // buildPython 的小版本必须与 App 的 Python 版本一致。
        // CI 通过 CHAQUOPY_BUILD_PYTHON 指向 actions/setup-python 安装的 3.14。
        buildPython(chaquopyBuildPython)
    }
}

dependencies {
  val composeBom = platform(libs.androidx.compose.bom)
  implementation(composeBom)
  androidTestImplementation(composeBom)

  // Core Android dependencies
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.lifecycle.runtime.ktx)
  implementation(libs.androidx.activity.compose)

  // Arch Components
  implementation(libs.androidx.lifecycle.runtime.compose)
  implementation(libs.androidx.lifecycle.viewmodel.compose)

  // Compose
  implementation(libs.androidx.compose.ui)
  implementation(libs.androidx.compose.ui.tooling.preview)
  implementation(libs.androidx.compose.material3)
  // Tooling
  debugImplementation(libs.androidx.compose.ui.tooling)
  // Instrumented tests
  androidTestImplementation(libs.androidx.compose.ui.test.junit4)
  debugImplementation(libs.androidx.compose.ui.test.manifest)

  // Local tests: jUnit, coroutines
  testImplementation(libs.junit)
  testImplementation(libs.kotlinx.coroutines.test)

  // Instrumented tests: jUnit rules and runners
  androidTestImplementation(libs.androidx.test.core)
  androidTestImplementation(libs.androidx.test.ext.junit)
  androidTestImplementation(libs.androidx.test.runner)
  androidTestImplementation(libs.androidx.test.espresso.core)
}
