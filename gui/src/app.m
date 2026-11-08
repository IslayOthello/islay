// Native macOS Desktop Application Wrapper for Islay (WZebra Style)
#import <Cocoa/Cocoa.h>
#import <WebKit/WebKit.h>

@interface AppDelegate : NSObject <NSApplicationDelegate, WKScriptMessageHandler>
@property (strong, nonatomic) NSWindow *window;
@property (strong, nonatomic) WKWebView *webView;
@property (strong, nonatomic) NSTask *luaTask;
@property (strong, nonatomic) NSPipe *inPipe;
@property (strong, nonatomic) NSPipe *outPipe;
@property (strong, nonatomic) NSMutableData *readData;
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    // 1. Build and install Native macOS System Menubar
    [self setupNativeMainMenu];

    // 2. Configure Window with Light Appearance
    NSRect frame = NSMakeRect(120, 100, 1140, 740);
    self.window = [[NSWindow alloc] initWithContentRect:frame
        styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable)
        backing:NSBackingStoreBuffered defer:NO];
    [self.window setTitle:@"Islay — Othello Engine Studio"];
    [self.window setAppearance:[NSAppearance appearanceNamed:NSAppearanceNameAqua]]; // Force clean Light Theme
    [self.window setMinSize:NSMakeSize(960, 620)];

    // 3. Configure WebKit & Message Bridge
    WKWebViewConfiguration *config = [[WKWebViewConfiguration alloc] init];
    [config.userContentController addScriptMessageHandler:self name:@"islay"];

    self.webView = [[WKWebView alloc] initWithFrame:self.window.contentView.bounds configuration:config];
    self.webView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    [self.window.contentView addSubview:self.webView];

    // 4. Launch Lua Backend Subprocess
    [self startLuaBackend];

    // 5. Load Desktop Web Interface
    NSString *bundlePath = [[NSBundle mainBundle] bundlePath];
    NSString *htmlPath = [[bundlePath stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"gui/web/index.html"];
    if (![[NSFileManager defaultManager] fileExistsAtPath:htmlPath]) {
        // Fallback relative to current working directory
        htmlPath = [NSString stringWithFormat:@"%@/gui/web/index.html", [[NSFileManager defaultManager] currentDirectoryPath]];
    }

    NSURL *url = [NSURL fileURLWithPath:htmlPath];
    [self.webView loadFileURL:url allowingReadAccessToURL:[url URLByDeletingLastPathComponent]];

    [self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

- (void)setupNativeMainMenu {
    NSMenu *mainMenu = [[NSMenu alloc] init];

    // --- 1. Application Menu (Islay) ---
    NSMenuItem *appMenuItem = [[NSMenuItem alloc] init];
    NSMenu *appMenu = [[NSMenu alloc] initWithTitle:@"Islay"];
    [appMenu addItemWithTitle:@"About Islay" action:@selector(orderFrontStandardAboutPanel:) keyEquivalent:@""];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:@"Hide Islay" action:@selector(hide:) keyEquivalent:@"h"];
    NSMenuItem *hideOthers = [appMenu addItemWithTitle:@"Hide Others" action:@selector(hideOtherApplications:) keyEquivalent:@"h"];
    [hideOthers setKeyEquivalentModifierMask:(NSEventModifierFlagOption | NSEventModifierFlagCommand)];
    [appMenu addItemWithTitle:@"Show All" action:@selector(unhideAllApplications:) keyEquivalent:@""];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:@"Quit Islay" action:@selector(terminate:) keyEquivalent:@"q"];
    [appMenuItem setSubmenu:appMenu];
    [mainMenu addItem:appMenuItem];

    // --- 2. File Menu ---
    NSMenuItem *fileMenuItem = [[NSMenuItem alloc] init];
    NSMenu *fileMenu = [[NSMenu alloc] initWithTitle:@"File"];
    [self addItemToMenu:fileMenu title:@"New Game" key:@"n" mask:NSEventModifierFlagCommand js:@"sendCmd('new_game')"];
    [fileMenu addItem:[NSMenuItem separatorItem]];
    [self addItemToMenu:fileMenu title:@"Copy Position FEN" key:@"c" mask:(NSEventModifierFlagCommand | NSEventModifierFlagShift) js:@"copyFen()"];
    [self addItemToMenu:fileMenu title:@"Paste Position FEN" key:@"v" mask:(NSEventModifierFlagCommand | NSEventModifierFlagShift) js:@"pasteFen()"];
    [fileMenu addItem:[NSMenuItem separatorItem]];
    [fileMenu addItemWithTitle:@"Close Window" action:@selector(performClose:) keyEquivalent:@"w"];
    [fileMenuItem setSubmenu:fileMenu];
    [mainMenu addItem:fileMenuItem];

    // --- 3. Position Menu ---
    NSMenuItem *posMenuItem = [[NSMenuItem alloc] init];
    NSMenu *posMenu = [[NSMenu alloc] initWithTitle:@"Position"];
    [self addItemToMenu:posMenu title:@"Edit Board Position" key:@"e" mask:0 js:@"switchTab('setup')"];
    [posMenu addItem:[NSMenuItem separatorItem]];
    [self addItemToMenu:posMenu title:@"Invert Colors" key:@"" mask:0 js:@"sendCmd('setup_invert')"];
    [self addItemToMenu:posMenu title:@"Clear Board" key:@"" mask:0 js:@"sendCmd('setup_clear')"];
    [self addItemToMenu:posMenu title:@"Reset Standard Startpos" key:@"" mask:0 js:@"sendCmd('setup_startpos')"];
    [posMenuItem setSubmenu:posMenu];
    [mainMenu addItem:posMenuItem];

    // --- 4. Game Menu ---
    NSMenuItem *gameMenuItem = [[NSMenuItem alloc] init];
    NSMenu *gameMenu = [[NSMenu alloc] initWithTitle:@"Game"];
    [self addItemToMenu:gameMenu title:@"Play Black vs Islay" key:@"" mask:0 js:@"setMode(1)"];
    [self addItemToMenu:gameMenu title:@"Play White vs Islay" key:@"" mask:0 js:@"setMode(2)"];
    [self addItemToMenu:gameMenu title:@"Two Players (Manual)" key:@"" mask:0 js:@"setMode(3)"];
    [self addItemToMenu:gameMenu title:@"Self-Play (Engine vs Engine)" key:@"" mask:0 js:@"setMode(4)"];
    [gameMenu addItem:[NSMenuItem separatorItem]];
    [self addItemToMenu:gameMenu title:@"Takeback Move" key:@"z" mask:NSEventModifierFlagCommand js:@"sendCmd('undo')"];
    [self addItemToMenu:gameMenu title:@"Step Forward" key:@"z" mask:(NSEventModifierFlagCommand | NSEventModifierFlagShift) js:@"sendCmd('redo')"];
    [self addItemToMenu:gameMenu title:@"Pass Turn" key:@" " mask:0 js:@"sendCmd('pass')"];
    [gameMenuItem setSubmenu:gameMenu];
    [mainMenu addItem:gameMenuItem];

    // --- 5. Engine Menu ---
    NSMenuItem *engMenuItem = [[NSMenuItem alloc] init];
    NSMenu *engMenu = [[NSMenu alloc] initWithTitle:@"Engine"];
    [self addItemToMenu:engMenu title:@"Suggest Best Move" key:@"h" mask:0 js:@"sendCmd('hint')"];
    [self addItemToMenu:engMenu title:@"Live Search Tree" key:@"" mask:0 js:@"switchTab('analysis')"];
    [engMenu addItem:[NSMenuItem separatorItem]];
    [self addItemToMenu:engMenu title:@"Perft Movegen Benchmark" key:@"p" mask:0 js:@"triggerPerft()"];
    [engMenuItem setSubmenu:engMenu];
    [mainMenu addItem:engMenuItem];

    // --- 6. View Menu ---
    NSMenuItem *viewMenuItem = [[NSMenuItem alloc] init];
    NSMenu *viewMenu = [[NSMenu alloc] initWithTitle:@"View"];
    [self addItemToMenu:viewMenu title:@"Move Numbers on Discs" key:@"1" mask:NSEventModifierFlagCommand js:@"toggleNumbers()"];
    [self addItemToMenu:viewMenu title:@"Candidate Evaluations" key:@"2" mask:NSEventModifierFlagCommand js:@"toggleEvals()"];
    [self addItemToMenu:viewMenu title:@"Advantage Trajectory Curve" key:@"3" mask:NSEventModifierFlagCommand js:@"toggleGraph()"];
    [viewMenu addItem:[NSMenuItem separatorItem]];
    [viewMenu addItemWithTitle:@"Toggle Full Screen" action:@selector(toggleFullScreen:) keyEquivalent:@"f"];
    [viewMenuItem setSubmenu:viewMenu];
    [mainMenu addItem:viewMenuItem];

    // --- 7. Options Menu ---
    NSMenuItem *optMenuItem = [[NSMenuItem alloc] init];
    NSMenu *optMenu = [[NSMenu alloc] initWithTitle:@"Options"];

    NSMenuItem *rulesItem = [[NSMenuItem alloc] initWithTitle:@"Game Rule" action:nil keyEquivalent:@""];
    NSMenu *rulesSub = [[NSMenu alloc] initWithTitle:@"Game Rule"];
    [self addItemToMenu:rulesSub title:@"Othello (Pass allowed)" key:@"" mask:0 js:@"setRule('Othello')"];
    [self addItemToMenu:rulesSub title:@"Reversi (Stops on no move)" key:@"" mask:0 js:@"setRule('Reversi')"];
    [rulesItem setSubmenu:rulesSub];
    [optMenu addItem:rulesItem];

    NSMenuItem *budgetItem = [[NSMenuItem alloc] initWithTitle:@"Search Budget" action:nil keyEquivalent:@""];
    NSMenu *budgetSub = [[NSMenu alloc] initWithTitle:@"Search Budget"];
    [self addItemToMenu:budgetSub title:@"Casual (1K nodes)" key:@"" mask:0 js:@"setBudget(1)"];
    [self addItemToMenu:budgetSub title:@"Club (10K nodes)" key:@"" mask:0 js:@"setBudget(2)"];
    [self addItemToMenu:budgetSub title:@"Master (50K nodes)" key:@"" mask:0 js:@"setBudget(3)"];
    [self addItemToMenu:budgetSub title:@"Timed (500 ms)" key:@"" mask:0 js:@"setBudget(4)"];
    [budgetItem setSubmenu:budgetSub];
    [optMenu addItem:budgetItem];

    [optMenuItem setSubmenu:optMenu];
    [mainMenu addItem:optMenuItem];

    // --- 8. Window Menu ---
    NSMenuItem *winMenuItem = [[NSMenuItem alloc] init];
    NSMenu *winMenu = [[NSMenu alloc] initWithTitle:@"Window"];
    [winMenu addItemWithTitle:@"Minimize" action:@selector(performMiniaturize:) keyEquivalent:@"m"];
    [winMenu addItemWithTitle:@"Zoom" action:@selector(performZoom:) keyEquivalent:@""];
    [winMenu addItem:[NSMenuItem separatorItem]];
    [winMenu addItemWithTitle:@"Bring All to Front" action:@selector(arrangeInFront:) keyEquivalent:@""];
    [winMenuItem setSubmenu:winMenu];
    [mainMenu addItem:winMenuItem];
    [NSApp setWindowsMenu:winMenu];

    [NSApp setMainMenu:mainMenu];
}

- (NSMenuItem *)addItemToMenu:(NSMenu *)menu title:(NSString *)title key:(NSString *)key mask:(NSEventModifierFlags)mask js:(NSString *)js {
    NSMenuItem *item = [[NSMenuItem alloc] initWithTitle:title action:@selector(menuActionTriggered:) keyEquivalent:key];
    [item setTarget:self];
    [item setRepresentedObject:js];
    if (mask != 0) {
        [item setKeyEquivalentModifierMask:mask];
    }
    [menu addItem:item];
    return item;
}

- (void)menuActionTriggered:(NSMenuItem *)sender {
    NSString *js = (NSString *)[sender representedObject];
    if (js && self.webView) {
        [self.webView evaluateJavaScript:js completionHandler:nil];
    }
}

- (void)startLuaBackend {
    self.inPipe = [NSPipe pipe];
    self.outPipe = [NSPipe pipe];
    self.readData = [NSMutableData data];

    self.luaTask = [[NSTask alloc] init];
    
    // Check luajit in standard paths
    NSString *luajitPath = @"/opt/homebrew/bin/luajit";
    if (![[NSFileManager defaultManager] fileExistsAtPath:luajitPath]) {
        luajitPath = @"/usr/local/bin/luajit";
    }
    if (![[NSFileManager defaultManager] fileExistsAtPath:luajitPath]) {
        luajitPath = @"/usr/bin/lua";
    }

    [self.luaTask setLaunchPath:luajitPath];
    [self.luaTask setArguments:@[@"gui/bridge.lua", @"pipe"]];
    [self.luaTask setStandardInput:self.inPipe];
    [self.luaTask setStandardOutput:self.outPipe];

    NSFileHandle *outHandle = [self.outPipe fileHandleForReading];
    __weak typeof(self) weakSelf = self;
    outHandle.readabilityHandler = ^(NSFileHandle *handle) {
        NSData *data = [handle availableData];
        if (data.length > 0) {
            [weakSelf processBackendOutput:data];
        }
    };

    @try {
        [self.luaTask launch];
    } @catch (NSException *e) {
        NSLog(@"Failed to launch Lua backend: %@", e);
    }
}

- (void)processBackendOutput:(NSData *)data {
    [self.readData appendData:data];
    
    const char *bytes = (const char *)[self.readData bytes];
    NSUInteger length = [self.readData length];
    NSUInteger start = 0;

    for (NSUInteger i = 0; i < length; i++) {
        if (bytes[i] == '\n') {
            NSData *lineData = [self.readData subdataWithRange:NSMakeRange(start, i - start)];
            NSString *line = [[NSString alloc] initWithData:lineData encoding:NSUTF8StringEncoding];
            if (line.length > 0) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    [self dispatchToWebView:line];
                });
            }
            start = i + 1;
        }
    }

    if (start > 0) {
        [self.readData replaceBytesInRange:NSMakeRange(0, start) withBytes:NULL length:0];
    }
}

- (void)dispatchToWebView:(NSString *)jsonString {
    NSString *escaped = [jsonString stringByReplacingOccurrencesOfString:@"\\" withString:@"\\\\"];
    escaped = [escaped stringByReplacingOccurrencesOfString:@"'" withString:@"\\'"];
    NSString *js = [NSString stringWithFormat:@"window.onIslayMessage('%@');", escaped];
    [self.webView evaluateJavaScript:js completionHandler:nil];
}

// Receive JavaScript Messages from the UI
- (void)userContentController:(WKUserContentController *)userContentController didReceiveScriptMessage:(WKScriptMessage *)message {
    if ([message.body isKindOfClass:[NSString class]]) {
        NSString *msg = (NSString *)message.body;
        NSData *data = [[msg stringByAppendingString:@"\n"] dataUsingEncoding:NSUTF8StringEncoding];
        @try {
            [[self.inPipe fileHandleForWriting] writeData:data];
        } @catch (NSException *e) {
            NSLog(@"Write to Lua backend failed: %@", e);
        }
    }
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return YES;
}

- (void)applicationWillTerminate:(NSNotification *)aNotification {
    if (self.luaTask && [self.luaTask isRunning]) {
        [self.luaTask terminate];
    }
}

@end

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        AppDelegate *delegate = [[AppDelegate alloc] init];
        [app setDelegate:delegate];
        [app run];
    }
    return 0;
}
