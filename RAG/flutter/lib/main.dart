import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});


  @override

  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false, // 우상단 'Debug' 띠 숨기기
      home: Scaffold(
        body: Center(
          child: Text(
            '플러터 개발환경 설정 완료',
            style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
                color: Colors.blue,
            ),
          ),
        ),
      ),
    );
  }
}