import React from 'react';
import type { CredibilityResult } from '../types';

interface CredibilityMeterProps {
    credibility?: CredibilityResult;
    isLoading?: boolean;
}

export const CredibilityMeter: React.FC<CredibilityMeterProps> = ({ credibility, isLoading }) => {
    if (isLoading) {
        return (
            <div className="bg-white rounded-3xl border border-[#E2E8F0] shadow-sm p-6 animate-pulse">
                <div className="h-5 bg-slate-200 rounded w-1/3 mb-4" />
                <div className="h-20 bg-slate-100 rounded-2xl mb-3" />
                <div className="h-4 bg-slate-200 rounded w-2/3" />
            </div>
        );
    }

    if (!credibility) {
        return null;
    }

    const { credibilityScore, fakeProbability, riskLevel, flaggedSignals } = credibility;
    const credPercent = Math.round(credibilityScore * 100);
    const fakePercent = Math.round(fakeProbability * 100);

    const getRiskConfig = () => {
        switch (riskLevel) {
            case 'HIGH':
                return {
                    title: 'High Risk Fake / Spam Review',
                    subtitle: 'Strong indicators of bot generation, deceptive phrasing, or promotional spam.',
                    color: '#EF4444',
                    bgColor: '#FEF2F2',
                    borderColor: '#FCA5A5',
                    badgeBg: '#FEE2E2',
                    badgeText: '#991B1B',
                    icon: 'gpp_bad',
                };
            case 'MEDIUM':
                return {
                    title: 'Suspicious / Moderately Polarized',
                    subtitle: 'Elevated promotional superlatives or formatting anomalies detected.',
                    color: '#F59E0B',
                    bgColor: '#FFFBEB',
                    borderColor: '#FCD34D',
                    badgeBg: '#FEF3C7',
                    badgeText: '#92400E',
                    icon: 'warning',
                };
            default:
                return {
                    title: 'Authentic Customer Review',
                    subtitle: 'Organic language structure and natural syntax consistent with genuine buyer feedback.',
                    color: '#10B981',
                    bgColor: '#F0FDF4',
                    borderColor: '#86EFAC',
                    badgeBg: '#DCFCE7',
                    badgeText: '#166534',
                    icon: 'verified_user',
                };
        }
    };

    const config = getRiskConfig();

    return (
        <div className="bg-white rounded-3xl border border-[#E2E8F0] shadow-sm p-6 transition-all">
            {/* Header */}
            <div className="flex items-center justify-between mb-5">
                <div className="flex items-center gap-2.5">
                    <div
                        className="w-9 h-9 rounded-xl flex items-center justify-center font-bold"
                        style={{ backgroundColor: config.badgeBg, color: config.color }}
                    >
                        <span className="material-symbols-outlined text-[20px]">{config.icon}</span>
                    </div>
                    <div>
                        <h3 className="text-base font-bold text-[#0F172A]">Review Authenticity Assessment</h3>
                        <p className="text-xs text-[#64748B]">Credibility scoring and spam detection engine</p>
                    </div>
                </div>

                <span
                    className="text-xs font-extrabold px-3 py-1 rounded-full uppercase tracking-wider border"
                    style={{
                        backgroundColor: config.badgeBg,
                        color: config.badgeText,
                        borderColor: config.borderColor,
                    }}
                >
                    {riskLevel} RISK
                </span>
            </div>

            {/* Authenticity Banner Card */}
            <div
                className="rounded-2xl p-4.5 border mb-5 transition-all"
                style={{
                    backgroundColor: config.bgColor,
                    borderColor: config.borderColor,
                }}
            >
                <div className="flex items-center justify-between gap-4">
                    <div>
                        <h4 className="text-sm font-extrabold text-[#0F172A]">{config.title}</h4>
                        <p className="text-xs text-[#475569] mt-0.5 leading-relaxed">{config.subtitle}</p>
                    </div>
                    <div className="text-right shrink-0">
                        <span
                            className="text-2xl font-black tracking-tight"
                            style={{ color: config.color }}
                        >
                            {credPercent}%
                        </span>
                        <p className="text-[10px] text-[#64748B] font-bold uppercase tracking-wider">
                            Credibility
                        </p>
                    </div>
                </div>

                {/* Authenticity Bar Gauge */}
                <div className="mt-3.5 space-y-1">
                    <div className="relative w-full h-2.5 bg-black/10 rounded-full overflow-hidden">
                        <div
                            className="h-full rounded-full transition-all duration-700 ease-out"
                            style={{
                                width: `${credPercent}%`,
                                backgroundColor: config.color,
                            }}
                        />
                    </div>
                    <div className="flex justify-between text-[10px] text-[#64748B] font-bold pt-0.5">
                        <span>0% (Fake/Spam)</span>
                        <span>50%</span>
                        <span>100% (Genuine)</span>
                    </div>
                </div>
            </div>

            {/* Explainable Risk Signals */}
            <div>
                <span className="text-[11px] font-bold text-[#64748B] uppercase tracking-wider block mb-2">
                    Linguistic & Syntactic Risk Signals
                </span>

                {flaggedSignals.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                        {flaggedSignals.map((signal, idx) => (
                            <span
                                key={idx}
                                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-[#FEF2F2] text-[#991B1B] border border-[#FECACA]"
                            >
                                <span className="material-symbols-outlined text-[15px] text-[#DC2626]">
                                    report_problem
                                </span>
                                <span>{signal}</span>
                            </span>
                        ))}
                    </div>
                ) : (
                    <div className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold bg-[#F0FDF4] text-[#166534] border border-[#BBF7D0]">
                        <span className="material-symbols-outlined text-[16px] text-[#16A34A]">
                            check_circle
                        </span>
                        <span>Natural review syntax & zero suspicious linguistic anomalies detected.</span>
                    </div>
                )}
            </div>
        </div>
    );
};
