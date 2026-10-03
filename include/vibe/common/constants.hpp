#pragma once
/// @file constants.hpp
/// @brief 物理常数与数值常数。
///
/// 取值与 COSMO/WRF 技术文档一致，便于与业务模式对比：
///   [D15] Baldauf et al. (2011) COSMO 描述
///   [D16] Skamarock et al. (2008) WRF ARW 技术说明
///   [P9] Monin-Obukhov 相似理论常数
///   [O9] WMO-No.8 观测规范

#include "vibe/common/types.hpp"

namespace vibe {

// ---------------------------- 干空气热力学 --------------------------------
inline constexpr Real kGravity      = Real(9.80665);      ///< 重力加速度 m/s^2
inline constexpr Real kRd           = Real(287.05);       ///< 干空气气体常数 J/(kg K)
inline constexpr Real kRv           = Real(461.51);       ///< 水汽气体常数 J/(kg K)
inline constexpr Real kCp           = Real(1004.64);      ///< 定压比热 J/(kg K)
inline constexpr Real kCv           = Real(717.63);       ///< 定容比热 J/(kg K)
inline constexpr Real kKappa        = Real(0.28571);      ///< Rd/cp
inline constexpr Real kEpsilonVap   = Real(0.62197);      ///< Rd/Rv（=epsilon）
inline constexpr Real kP0           = Real(100000.0);     ///< 参考气压 Pa
inline constexpr Real kT0           = Real(273.15);       ///< 0 摄氏度 K
inline constexpr Real kTtriple      = Real(273.16);       ///< 三相点 K
inline constexpr Real kGammaDry     = Real(1.4);          ///< cp/cv（干空气）

// ---------------------------- 水物质 --------------------------------------
inline constexpr Real kLv           = Real(2.501e6);      ///< 汽化潜热 J/kg（0 C）
inline constexpr Real kLs           = Real(2.834e6);      ///< 升华潜热 J/kg
inline constexpr Real kLf           = Real(3.337e5);      ///< 融化潜热 J/kg
inline constexpr Real kRhoWater     = Real(1000.0);       ///< 液态水密度 kg/m^3
inline constexpr Real kRhoIce       = Real(916.7);        ///< 冰密度 kg/m^3
inline constexpr Real kDvWater      = Real(2.21e-5);      ///< 水汽扩散系数 m^2/s
inline constexpr Real kThermCondAir = Real(2.40e-2);      ///< 空气导热系数 W/(m K)

// ---------------------------- 辐射 ----------------------------------------
inline constexpr Real kStefanBoltzmann = Real(5.670374419e-8);  ///< W/(m^2 K^4)
inline constexpr Real kSolarConstant   = Real(1361.0);          ///< W/m^2
inline constexpr Real kCO2ppm          = Real(420.0);
inline constexpr Real kOzoneColumnDU   = Real(300.0);

// ---------------------------- 地球 ----------------------------------------
inline constexpr Real kEarthRadius  = Real(6.371229e6);   ///< 平均半径 m
inline constexpr Real kOmega        = Real(7.2921159e-5); ///< 自转角速度 rad/s
inline constexpr Real kReferenceLat = Real(45.0);         ///< f-plane 参考纬度

// ---------------------------- 边界层/陆面 ---------------------------------
inline constexpr Real kVonKarman    = Real(0.40);         ///< von Karman 常数
inline constexpr Real kRoughnessLand  = Real(0.10);       ///< 陆地粗糙度 m
inline constexpr Real kRoughnessSea   = Real(1.0e-4);     ///< 海面粗糙度 m
inline constexpr Real kAlbedoSea      = Real(0.06);
inline constexpr Real kAlbedoLand     = Real(0.20);
inline constexpr Real kSoilHeatCap    = Real(2.0e6);      ///< J/(m^3 K)

// ---------------------------- 数值常数 ------------------------------------
inline constexpr Real kPi      = Real(3.14159265358979323846);
inline constexpr Real kTwoPi   = Real(6.28318530717958647692);
inline constexpr Real kDegToRad = kPi / Real(180.0);
inline constexpr Real kRadToDeg = Real(180.0) / kPi;
inline constexpr Real kSqrt2   = Real(1.41421356237309504880);

/// 判断是否为可降水量类水物质
inline constexpr bool is_hydrometeor(int species) noexcept { return species >= 1 && species <= 5; }

}  // namespace vibe
