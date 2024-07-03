# Found using code in findFrequencies.R
# Credit to Mitchell O'Hara-Wild and Rob J Hyndman for the R code
# Link: https://github.com/robjhyndman/forecast

frequencies = {
    'economics_1.csv': 7,
    'economics_10.csv': 1,
    'economics_100.csv': 6,
    'economics_11.csv': 2,
    'economics_12.csv': 4,
    'economics_13.csv': 1,
    'economics_14.csv': 1,
    'economics_15.csv': 1,
    'economics_16.csv': 1,
    'economics_17.csv': 12,
    'economics_18.csv': 12,
    'economics_19.csv': 6,
    'economics_2.csv': 6,
    'economics_20.csv': 6,
    'economics_21.csv': 499,
    'economics_22.csv': 12,
    'economics_23.csv': 12,
    'economics_24.csv': 4,
    'economics_25.csv': 4,
    'economics_26.csv': 2,
    'economics_27.csv': 1,
    'economics_28.csv': 12,
    'economics_29.csv': 6,
    'economics_3.csv': 1,
    'economics_30.csv': 12,
    'economics_31.csv': 12,
    'economics_32.csv': 53,
    'economics_33.csv': 12,
    'economics_34.csv': 333,
    'economics_35.csv': 4,
    'economics_36.csv': 12,
    'economics_37.csv': 12,
    'economics_38.csv': 1,
    'economics_39.csv': 125,
    'economics_4.csv': 1,
    'economics_40.csv': 12,
    'economics_41.csv': 12,
    'economics_42.csv': 12,
    'economics_43.csv': 12,
    'economics_44.csv': 12,
    'economics_45.csv': 6,
    'economics_46.csv': 4,
    'economics_47.csv': 1,
    'economics_48.csv': 4,
    'economics_49.csv': 1,
    'economics_5.csv': 8,
    'economics_50.csv': 4,
    'economics_51.csv': 2,
    'economics_52.csv': 1,
    'economics_53.csv': 1,
    'economics_54.csv': 1,
    'economics_55.csv': 1,
    'economics_56.csv': 6,
    'economics_57.csv': 1,
    'economics_58.csv': 6,
    'economics_59.csv': 4,
    'economics_6.csv': 1,
    'economics_60.csv': 6,
    'economics_61.csv': 4,
    'economics_62.csv': 1,
    'economics_63.csv': 12,
    'economics_64.csv': 1,
    'economics_65.csv': 12,
    'economics_66.csv': 1,
    'economics_67.csv': 12,
    'economics_68.csv': 1,
    'economics_69.csv': 12,
    'economics_7.csv': 6,
    'economics_70.csv': 4,
    'economics_71.csv': 4,
    'economics_72.csv': 4,
    'economics_73.csv': 2,
    'economics_74.csv': 4,
    'economics_75.csv': 4,
    'economics_76.csv': 4,
    'economics_77.csv': 4,
    'economics_78.csv': 4,
    'economics_79.csv': 4,
    'economics_8.csv': 1,
    'economics_80.csv': 4,
    'economics_81.csv': 4,
    'economics_82.csv': 2,
    'economics_83.csv': 2,
    'economics_84.csv': 2,
    'economics_85.csv': 2,
    'economics_86.csv': 2,
    'economics_87.csv': 4,
    'economics_88.csv': 4,
    'economics_89.csv': 4,
    'economics_9.csv': 8,
    'economics_90.csv': 12,
    'economics_91.csv': 12,
    'economics_92.csv': 6,
    'economics_93.csv': 6,
    'economics_94.csv': 1,
    'economics_95.csv': 77,
    'economics_96.csv': 6,
    'economics_97.csv': 4,
    'economics_98.csv': 4,
    'economics_99.csv': 4,
    'finance_1.csv': 1,
    'finance_10.csv': 22,
    'finance_100.csv': 1,
    'finance_11.csv': 8,
    'finance_12.csv': 1,
    'finance_13.csv': 1,
    'finance_14.csv': 12,
    'finance_15.csv': 12,
    'finance_16.csv': 1,
    'finance_17.csv': 11,
    'finance_18.csv': 13,
    'finance_19.csv': 1,
    'finance_2.csv': 1,
    'finance_20.csv': 1,
    'finance_21.csv': 14,
    'finance_22.csv': 9,
    'finance_23.csv': 1,
    'finance_24.csv': 1,
    'finance_25.csv': 1,
    'finance_26.csv': 1,
    'finance_27.csv': 4,
    'finance_28.csv': 7,
    'finance_29.csv': 6,
    'finance_3.csv': 1,
    'finance_30.csv': 12,
    'finance_31.csv': 333,
    'finance_32.csv': 200,
    'finance_33.csv': 6,
    'finance_34.csv': 1,
    'finance_35.csv': 1,
    'finance_36.csv': 6,
    'finance_37.csv': 2,
    'finance_38.csv': 4,
    'finance_39.csv': 2,
    'finance_4.csv': 3,
    'finance_40.csv': 4,
    'finance_41.csv': 4,
    'finance_42.csv': 1,
    'finance_43.csv': 4,
    'finance_44.csv': 13,
    'finance_45.csv': 14,
    'finance_46.csv': 16,
    'finance_47.csv': 17,
    'finance_48.csv': 3,
    'finance_49.csv': 1,
    'finance_5.csv': 1,
    'finance_50.csv': 7,
    'finance_51.csv': 4,
    'finance_52.csv': 1,
    'finance_53.csv': 1,
    'finance_54.csv': 3,
    'finance_55.csv': 2,
    'finance_56.csv': 1,
    'finance_57.csv': 1,
    'finance_58.csv': 16,
    'finance_59.csv': 1,
    'finance_6.csv': 12,
    'finance_60.csv': 166,
    'finance_61.csv': 1,
    'finance_62.csv': 10,
    'finance_63.csv': 1,
    'finance_64.csv': 1,
    'finance_65.csv': 12,
    'finance_66.csv': 16,
    'finance_67.csv': 1,
    'finance_68.csv': 10,
    'finance_69.csv': 1,
    'finance_7.csv': 7,
    'finance_70.csv': 10,
    'finance_71.csv': 12,
    'finance_72.csv': 15,
    'finance_73.csv': 1,
    'finance_74.csv': 1,
    'finance_75.csv': 3,
    'finance_76.csv': 7,
    'finance_77.csv': 7,
    'finance_78.csv': 1,
    'finance_79.csv': 7,
    'finance_8.csv': 7,
    'finance_80.csv': 1,
    'finance_81.csv': 3,
    'finance_82.csv': 3,
    'finance_83.csv': 7,
    'finance_84.csv': 1,
    'finance_85.csv': 1,
    'finance_86.csv': 3,
    'finance_87.csv': 6,
    'finance_88.csv': 1,
    'finance_89.csv': 1,
    'finance_9.csv': 10,
    'finance_90.csv': 8,
    'finance_91.csv': 4,
    'finance_92.csv': 1,
    'finance_93.csv': 1,
    'finance_94.csv': 3,
    'finance_95.csv': 3,
    'finance_96.csv': 1,
    'finance_97.csv': 3,
    'finance_98.csv': 4,
    'finance_99.csv': 6,
    'human_1.csv': 12,
    'human_10.csv': 143,
    'human_100.csv': 10,
    'human_11.csv': 8,
    'human_12.csv': 1,
    'human_13.csv': 7,
    'human_14.csv': 200,
    'human_15.csv': 12,
    'human_16.csv': 1,
    'human_17.csv': 12,
    'human_18.csv': 3,
    'human_19.csv': 7,
    'human_2.csv': 24,
    'human_20.csv': 100,
    'human_21.csv': 24,
    'human_22.csv': 24,
    'human_23.csv': 24,
    'human_24.csv': 24,
    'human_25.csv': 24,
    'human_26.csv': 24,
    'human_27.csv': 24,
    'human_28.csv': 24,
    'human_29.csv': 24,
    'human_3.csv': 91,
    'human_30.csv': 125,
    'human_31.csv': 7,
    'human_32.csv': 12,
    'human_33.csv': 24,
    'human_34.csv': 6,
    'human_35.csv': 3,
    'human_36.csv': 12,
    'human_37.csv': 7,
    'human_38.csv': 1,
    'human_39.csv': 12,
    'human_4.csv': 1,
    'human_40.csv': 3,
    'human_41.csv': 24,
    'human_42.csv': 24,
    'human_43.csv': 24,
    'human_44.csv': 24,
    'human_45.csv': 24,
    'human_46.csv': 24,
    'human_47.csv': 24,
    'human_48.csv': 24,
    'human_49.csv': 24,
    'human_5.csv': 125,
    'human_50.csv': 24,
    'human_51.csv': 24,
    'human_52.csv': 24,
    'human_53.csv': 6,
    'human_54.csv': 100,
    'human_55.csv': 24,
    'human_56.csv': 6,
    'human_57.csv': 12,
    'human_58.csv': 12,
    'human_59.csv': 12,
    'human_6.csv': 333,
    'human_60.csv': 6,
    'human_61.csv': 12,
    'human_62.csv': 12,
    'human_63.csv': 6,
    'human_64.csv': 1,
    'human_65.csv': 24,
    'human_66.csv': 24,
    'human_67.csv': 24,
    'human_68.csv': 1,
    'human_69.csv': 3,
    'human_7.csv': 125,
    'human_70.csv': 2,
    'human_71.csv': 2,
    'human_72.csv': 2,
    'human_73.csv': 2,
    'human_74.csv': 3,
    'human_75.csv': 2,
    'human_76.csv': 6,
    'human_77.csv': 5,
    'human_78.csv': 8,
    'human_79.csv': 8,
    'human_8.csv': 143,
    'human_80.csv': 9,
    'human_81.csv': 7,
    'human_82.csv': 24,
    'human_83.csv': 24,
    'human_84.csv': 24,
    'human_85.csv': 24,
    'human_86.csv': 24,
    'human_87.csv': 24,
    'human_88.csv': 24,
    'human_89.csv': 24,
    'human_9.csv': 12,
    'human_90.csv': 24,
    'human_91.csv': 24,
    'human_92.csv': 24,
    'human_93.csv': 24,
    'human_94.csv': 24,
    'human_95.csv': 24,
    'human_96.csv': 24,
    'human_97.csv': 24,
    'human_98.csv': 12,
    'human_99.csv': 6,
    'nature_1.csv': 15,
    'nature_10.csv': 12,
    'nature_100.csv': 10,
    'nature_11.csv': 12,
    'nature_12.csv': 30,
    'nature_13.csv': 12,
    'nature_14.csv': 12,
    'nature_15.csv': 32,
    'nature_16.csv': 21,
    'nature_17.csv': 1,
    'nature_18.csv': 1,
    'nature_19.csv': 12,
    'nature_2.csv': 12,
    'nature_20.csv': 6,
    'nature_21.csv': 6,
    'nature_22.csv': 6,
    'nature_23.csv': 3,
    'nature_24.csv': 12,
    'nature_25.csv': 10,
    'nature_26.csv': 12,
    'nature_27.csv': 1,
    'nature_28.csv': 50,
    'nature_29.csv': 12,
    'nature_3.csv': 7,
    'nature_30.csv': 12,
    'nature_31.csv': 14,
    'nature_32.csv': 12,
    'nature_33.csv': 12,
    'nature_34.csv': 12,
    'nature_35.csv': 6,
    'nature_36.csv': 12,
    'nature_37.csv': 1,
    'nature_38.csv': 12,
    'nature_39.csv': 1,
    'nature_4.csv': 12,
    'nature_40.csv': 3,
    'nature_41.csv': 12,
    'nature_42.csv': 12,
    'nature_43.csv': 12,
    'nature_44.csv': 12,
    'nature_45.csv': 12,
    'nature_46.csv': 12,
    'nature_47.csv': 12,
    'nature_48.csv': 12,
    'nature_49.csv': 12,
    'nature_5.csv': 4,
    'nature_50.csv': 12,
    'nature_51.csv': 12,
    'nature_52.csv': 12,
    'nature_53.csv': 2,
    'nature_54.csv': 1,
    'nature_55.csv': 3,
    'nature_56.csv': 12,
    'nature_57.csv': 125,
    'nature_58.csv': 12,
    'nature_59.csv': 12,
    'nature_6.csv': 4,
    'nature_60.csv': 1,
    'nature_61.csv': 1,
    'nature_62.csv': 1,
    'nature_63.csv': 12,
    'nature_64.csv': 24,
    'nature_65.csv': 7,
    'nature_66.csv': 12,
    'nature_67.csv': 12,
    'nature_68.csv': 1,
    'nature_69.csv': 11,
    'nature_7.csv': 12,
    'nature_70.csv': 12,
    'nature_71.csv': 12,
    'nature_72.csv': 12,
    'nature_73.csv': 1,
    'nature_74.csv': 11,
    'nature_75.csv': 30,
    'nature_76.csv': 28,
    'nature_77.csv': 499,
    'nature_78.csv': 2,
    'nature_79.csv': 28,
    'nature_8.csv': 1,
    'nature_80.csv': 24,
    'nature_81.csv': 25,
    'nature_82.csv': 12,
    'nature_83.csv': 12,
    'nature_84.csv': 12,
    'nature_85.csv': 11,
    'nature_86.csv': 1,
    'nature_87.csv': 48,
    'nature_88.csv': 1,
    'nature_89.csv': 1,
    'nature_9.csv': 1,
    'nature_90.csv': 1,
    'nature_91.csv': 1,
    'nature_92.csv': 4,
    'nature_93.csv': 12,
    'nature_94.csv': 4,
    'nature_95.csv': 2,
    'nature_96.csv': 4,
    'nature_97.csv': 12,
    'nature_98.csv': 12,
    'nature_99.csv': 1,
}
